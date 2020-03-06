import json
import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src import FeatureExtractor, AnchorGenerator, Detector, GraphExtractor, makedirs
from src.constants import PARALLEL_ITERATIONS
from src.evaluation_numpy import EvaluatorNumpy
from src.input_pipeline import Pipeline
from src.input_pipeline.mean_teacher_pipeline import MeanTeacherPipeline
from src.result_util import output_prediction, output_feature_maps

CONFIG_PATH = 'config.json'


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        # load parameters
        self.params = json.load(open(CONFIG_PATH))
        self.model_params = self.params['model_params']
        self.input_params = self.params['input_pipeline_params']

        # input parameters
        self.gpu_num = args.gpu_num
        self.batch_size = args.batch_size_per_gpu * self.gpu_num
        self.image_size = self.input_params['image_size']

        # FaceBoxes model parameters
        self.epoch_source_only = self.model_params['epoch_source_only']
        self.source_max_iteration = int(self.model_params['source_max_iteration'] / (self.batch_size / 16))
        self.lr_boundaries = [int(self.model_params['lr_boundaries'][0] / self.gpu_num),
                              int(self.model_params['lr_boundaries'][1] / self.gpu_num)]
        self.localization_loss_weight = self.model_params['localization_loss_weight']
        self.classification_loss_weight = self.model_params['classification_loss_weight']
        self.weight_decay = self.model_params['weight_decay']

        # mean teacher parameters
        self.epoch_mean_teacher = self.model_params['epoch_mean_teacher']
        self.mean_teacher_lr = self.model_params['mean_teacher_lr']
        self.mt_lambda = self.model_params['mt_lambda']
        self.score_threshold = self.model_params['confidence_threshold']
        self.alpha = self.model_params['smooth_param_of_ema']

        # directory setting
        # log dir
        self.log_dir = os.path.join(args.log_dir, args.sub_dir, 'pretrain') \
            if args.phase == 'pretrain' else os.path.join(args.log_dir, args.sub_dir, 'train')
        makedirs(self.log_dir)

        # checkpoint dir
        self.pretrain_ckpt_dir = os.path.join(args.ckpt_dir, args.pretrain_ckpt_sub_dir)
        self.mt_ckpt_dir = os.path.join(args.ckpt_dir, args.sub_dir)
        makedirs(self.pretrain_ckpt_dir)
        makedirs(self.mt_ckpt_dir)

        # build graph
        self._build_model()

        # create writer and saver
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()

    def _build_model(self):
        # source input placeholder
        self.s_images = tf.placeholder(tf.float32, [self.batch_size, None, None, 3], name="s_images")
        self.s_boxes = tf.placeholder(tf.float32, [self.batch_size, None, 4], name="s_boxes")
        self.s_num_boxes = tf.placeholder(tf.int32, [self.batch_size], name="s_num_boxes")
        # divide source input
        self.s_images_per_gpu = tf.split(self.s_images, self.gpu_num)
        self.s_boxes_per_gpu = tf.split(self.s_boxes, self.gpu_num)
        self.s_num_boxes_per_gpu = tf.split(self.s_num_boxes, self.gpu_num)

        # target input placeholder
        self.t_images = tf.placeholder(tf.float32, [self.batch_size, None, None, 3], name="t_images")
        self.t_boxes = tf.placeholder(tf.float32, [self.batch_size, None, 4], name="t_boxes")
        self.t_num_boxes = tf.placeholder(tf.int32, [self.batch_size], name="t_num_boxes")
        # divide target input
        self.t_images_per_gpu = tf.split(self.t_images, self.gpu_num)
        self.t_boxes_per_gpu = tf.split(self.t_boxes, self.gpu_num)
        self.t_num_boxes_per_gpu = tf.split(self.t_num_boxes, self.gpu_num)

        # list to gather losses
        # self.s_supervised_total_loss_list = []
        self.consistency_loss_list = []

        # list to gather predictions
        self.s_prediction_list = []

        # each gpu model define
        for gpu_id in range(int(self.gpu_num)):
            reuse = (gpu_id > 0)
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    # student model
                    with tf.variable_scope('student'):
                        self.s_feature_extractor = FeatureExtractor(is_training=False)
                        self.s_anchor_generator = AnchorGenerator()
                        self.s_detector = Detector(self.s_images_per_gpu[gpu_id], self.s_feature_extractor,
                                                   self.s_anchor_generator)

                        with tf.name_scope('weight_decay'):
                            self.regularization_loss = add_weight_decay(self.weight_decay, scope='student')

                        with tf.name_scope('student_supervised_loss'):
                            s_labels = {'boxes': self.s_boxes_per_gpu[gpu_id],
                                        'num_boxes': self.s_num_boxes_per_gpu[gpu_id]}
                            s_losses = self.s_detector.loss(s_labels, self.model_params)
                            self.s_localization_loss = self.localization_loss_weight * s_losses['localization_loss']
                            self.s_classification_loss = \
                                self.classification_loss_weight * s_losses['classification_loss']
                            self.s_supervised_total_loss = \
                                self.s_localization_loss + self.s_classification_loss + self.regularization_loss
                            # self.s_supervised_total_loss_list.append(self.s_supervised_total_loss)
                            tf.add_to_collection('s_supervised_total_loss', self.s_supervised_total_loss)

                        with tf.name_scope('student_prediction'):
                            s_prediction = self.s_detector.get_predictions(
                                score_threshold=self.model_params['score_threshold'],
                                iou_threshold=self.model_params['iou_threshold'],
                                max_boxes=self.model_params['max_boxes']
                            )
                            self.s_prediction_list.append(s_prediction)

                    # teacher model
                    with tf.variable_scope('teacher'):
                        self.t_feature_extractor = FeatureExtractor(is_training=False)
                        self.t_anchor_generator = AnchorGenerator()
                        self.t_detector = Detector(self.t_images_per_gpu[gpu_id], self.t_feature_extractor,
                                                   self.t_anchor_generator)

                    # extract graph for consistency loss
                    self.s_class_pred = self.s_detector.get_class_prediction()
                    self.s_boxes_pred = self.s_detector.get_box_prediction()
                    self.t_class_pred = self.t_detector.get_class_prediction()
                    self.t_boxes_pred = self.t_detector.get_box_prediction()
                    self.s_selected_class, self.s_selected_boxes, self.t_selected_class, self.t_selected_boxes, _ \
                        = GraphExtractor(self.s_class_pred, self.s_boxes_pred, self.t_class_pred, self.t_boxes_pred)

                    with tf.name_scope('Region-level-Consistency-Loss'):
                        self.s_graph, _, self.t_graph, _, num_boxes = \
                            GraphExtractor(self.s_selected_class, self.s_selected_boxes, self.t_selected_class,
                                           self.t_selected_boxes, iou_use=False, score_threshold=self.score_threshold)

                        def RLC_fn(x):
                            s_g, t_g, num_box = x
                            return tf.reduce_mean((s_g[:num_box] - t_g[:num_box]) ** 2)

                        self.loss_RCL_per_image = tf.map_fn(RLC_fn, [self.s_graph, self.t_graph, num_boxes],
                                                            dtype=tf.float32, parallel_iterations=PARALLEL_ITERATIONS,
                                                            back_prop=True, swap_memory=False, infer_shape=True)
                        self.loss_RCL = tf.reduce_mean(self.loss_RCL_per_image)

                    with tf.name_scope('intEr-Graph-consistency-Loss'):
                        self.s_feat = tf.concat([self.s_selected_boxes,
                                                 tf.expand_dims(self.s_selected_class, axis=2)], axis=2)
                        self.t_feat = tf.concat([self.t_selected_boxes,
                                                 tf.expand_dims(self.t_selected_class, axis=2)], axis=2)
                        # make affinity matrix (cosine similarity)
                        self.s_AM = make_cosine_similarity_matrix(self.s_feat)
                        self.t_AM = make_cosine_similarity_matrix(self.t_feat)
                        self.loss_EGL = tf.reduce_mean(tf.reduce_mean((self.s_AM - self.t_AM) ** 2, axis=(1, 2)))

                    with tf.name_scope('intrA-Graph-consistency-Loss'):
                        self.loss_AGL = tf.reduce_mean(
                            tf.reduce_mean((tf.ones_like(self.s_AM) - self.s_AM), axis=(1, 2)))

                    # total consistency loss
                    self.loss_cons = self.loss_RCL + self.loss_EGL + self.loss_AGL
                    self.consistency_loss_list.append(self.loss_cons)

        # compute all loss over gpu
        self.s_total_loss = tf.reduce_mean(tf.get_collection('s_supervised_total_loss'))
        # self.s_total_loss = tf.reduce_mean(tf.stack(self.s_supervised_total_loss_list, axis=0))
        self.total_cons_loss = tf.reduce_mean(tf.stack(self.consistency_loss_list, axis=0))

        # learning rate for supervised optimizer
        with tf.variable_scope('learning_rate'):
            self.global_step = tf.train.get_global_step()
            if self.global_step is None:
                self.global_step = tf.train.create_global_step()
            self.source_learning_rate = tf.train.piecewise_constant(self.global_step, self.lr_boundaries,
                                                                    self.model_params['lr_values'])

        # getting variables
        self.student_vars = tf.trainable_variables(scope='student')
        self.teacher_vars = tf.trainable_variables(scope='teacher')
        print('----- student model variables -----')
        for var in self.student_vars:
            print(var)
        print('----- teacher model variables -----')
        for var in self.teacher_vars:
            print(var.name)

        # this optimizer is used with only supervised training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='student')
        with tf.control_dependencies(update_ops):
            self.s_optim_with_supervision = \
                tf.train.MomentumOptimizer(self.source_learning_rate, momentum=0.9, use_nesterov=True) \
                    .minimize(self.s_total_loss, global_step=self.global_step,
                              colocate_gradients_with_ops=True, var_list=self.student_vars)
        # this optimizer is used with mean teacher training
        self.mean_teacher_loss = self.mt_lambda * self.total_cons_loss
        self.s_optim_with_MT = tf.train.MomentumOptimizer(self.mean_teacher_lr, momentum=0.9, use_nesterov=True) \
            .minimize(self.mean_teacher_loss, global_step=self.global_step,
                      colocate_gradients_with_ops=True, var_list=self.student_vars)

        # summary
        regularization_loss_sum = tf.summary.scalar('regularization_loss', self.regularization_loss)
        s_localization_loss_sum = tf.summary.scalar('localization_loss', self.s_localization_loss)
        s_classification_loss_sum = tf.summary.scalar('classification_loss', self.s_classification_loss)
        s_supervised_total_loss_sum = tf.summary.scalar('s_total_loss', self.s_total_loss)
        s_lr_sum = tf.summary.scalar('learning_rate', self.source_learning_rate)
        self.merged_source_sum = tf.summary.merge([s_lr_sum, s_supervised_total_loss_sum, regularization_loss_sum,
                                                   s_localization_loss_sum, s_classification_loss_sum])

        loss_RCL_sum = tf.summary.scalar('RCL_loss', self.loss_RCL)
        loss_EGL_sum = tf.summary.scalar('EGL_loss', self.loss_EGL)
        loss_AGL_sum = tf.summary.scalar('AGL_loss', self.loss_AGL)
        total_cons_loss_sum = tf.summary.scalar('total_cons_loss', self.total_cons_loss)
        self.merged_MT_sum = tf.summary.merge([loss_RCL_sum, loss_EGL_sum,
                                               loss_AGL_sum, total_cons_loss_sum])

        # teacher weights update
        self.mt_initial_op = tf.group([tf.assign(t_var, s_var)
                                       for s_var, t_var in zip(self.student_vars, self.teacher_vars)])
        self.mt_update_op = tf.group([tf.assign(t_var, self.alpha * t_var + (1. - self.alpha) * s_var)
                                      for s_var, t_var in zip(self.student_vars, self.teacher_vars)])

    def pretrain(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # source training input
        train_next_el = self.source_input_fn(is_training=True)
        # val_next_el, num_val_data = self.source_input_fn(is_training=False)
        # evaluator = EvaluatorNumpy()

        print('start pretraining')
        # pretrain source images
        for idx in tqdm(range(self.source_max_iteration)):
            images, boxes, num_boxes, filenames = self.sess.run(train_next_el)
            feed_dict = {
                self.s_images: images,
                self.s_boxes: boxes,
                self.s_num_boxes: num_boxes,
            }
            _, summary = self.sess.run([self.s_optim_with_supervision, self.merged_source_sum], feed_dict=feed_dict)
            self.writer.add_summary(summary, idx)

        # save pretrain model
        self.save(step=self.source_max_iteration, is_pretrain=True, model_name='pretrain.model')
        print(' [*] saved model')

    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(is_pretrain=True):
            print(" [*] Load SUCCESS")
        else:
            assert False, " [!] Load failed..."

        # mean teacher training input
        source_train_next_el, num_source_train_data = self.source_input_fn(is_training=True)
        target_train_next_el, num_target_train_data = self.target_input_fn(is_training=True)
        # target_val_next_el, num_target_val_data = self.target_input_fn(is_training=False)
        # evaluator = EvaluatorNumpy()

        # initialize teacher weights
        self.sess.run(self.mt_initial_op)

        global_step = num_source_train_data + self.epoch_source_only
        num_step = num_target_train_data * self.epoch_mean_teacher
        for idx in tqdm(range(num_step)):
            # supervised update
            images, boxes, num_boxes, filenames = self.sess.run(source_train_next_el)
            feed_dict = {
                self.s_images: images,
                self.s_boxes: boxes,
                self.s_num_boxes: num_boxes,
            }
            _, summary = self.sess.run([self.s_optim_with_supervision, self.merged_source_sum], feed_dict=feed_dict)
            self.writer.add_summary(summary, global_step + idx)

            # unsupervised update
            images, boxes, num_boxes, filenames = self.sess.run(target_train_next_el)
            feed_dict = {
                self.s_images: images[:, 0, :],
                self.s_boxes: boxes[:, 0, :],
                self.s_num_boxes: num_boxes,
                self.t_images: images[:, 1, :],
                self.t_boxes: boxes[:, 1, :],
                self.t_num_boxes: num_boxes,
            }
            _, summary = self.sess.run([self.s_optim_with_MT, self.merged_MT_sum], feed_dict=feed_dict)
            self.writer.add_summary(summary, global_step + idx)

            # teacher weights update
            self.sess.run(self.mt_update_op)

            # if idx + 1 % 500 == 0:
            #     self.val(evaluator, target_val_next_el, global_step + idx)

        # save mean teacher model
        self.save(step=num_step, is_pretrain=False, model_name='MeanTeacher.model')
        print(' [*] saved model')

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if args.source_only_test:
            # source only training model loading
            if self.load(is_pretrain=True):
                print(" [*] Load SUCCESS")
            else:
                assert False, " [!] Load failed..."

            # val input_pipeline
            target_val_next_el, num_target_val_data = self.source_input_fn(is_training=False)
            # target_val_next_el, num_target_val_data = self.target_input_fn(is_training=False)
            print('num_val_source {}'.format(num_target_val_data))
            # evaluator
            evaluator = EvaluatorNumpy()

            # ----- source only -----
            total_ap_source_only = 0
            for idx in tqdm(range(num_target_val_data)):
                images, boxes, num_boxes, filenames = self.sess.run(target_val_next_el)
                feed_dict = {
                    self.s_images: images,
                    self.s_boxes: boxes,
                    self.s_num_boxes: num_boxes,
                }

                # prediction
                prediction = self.s_detector.get_predictions(
                    score_threshold=self.model_params['score_threshold'],
                    iou_threshold=self.model_params['iou_threshold'],
                    max_boxes=self.model_params['max_boxes']
                )
                prediction = self.sess.run(prediction, feed_dict=feed_dict)

                # evaluation
                val_labels = {'boxes': boxes, 'num_boxes': num_boxes}
                eval_result = evaluator.get_metric(filenames, val_labels, prediction)

                # sum ap
                total_ap_source_only += eval_result['AP']
            mAP_source_only = total_ap_source_only / num_target_val_data

            print('mAP_source_only: {}'.format(mAP_source_only))

        if args.MT_test:
            # mean teacher training model loading
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                assert False, " [!] Load failed..."

            # val input_pipeline
            # target_val_next_el, num_target_val_data = self.target_input_fn(is_training=False)
            target_val_next_el, num_target_val_data = self.source_input_fn(is_training=False)
            print('num_val_source {}'.format(num_target_val_data))
            # evaluator
            evaluator = EvaluatorNumpy()

            # ----- mean teacher -----
            total_ap_mean_teacher = 0
            for idx in tqdm(range(num_target_val_data)):
                images, boxes, num_boxes, filenames = self.sess.run(target_val_next_el)
                feed_dict = {
                    self.s_images: images,
                    self.s_boxes: boxes,
                    self.s_num_boxes: num_boxes,
                }

                # prediction
                prediction = self.s_detector.get_predictions(
                    score_threshold=self.model_params['score_threshold'],
                    iou_threshold=self.model_params['iou_threshold'],
                    max_boxes=self.model_params['max_boxes']
                )
                prediction = self.sess.run(prediction, feed_dict=feed_dict)

                # evaluation
                val_labels = {'boxes': boxes, 'num_boxes': num_boxes}
                eval_result = evaluator.get_metric(filenames, val_labels, prediction)

                # sum ap
                total_ap_mean_teacher += eval_result['AP']
            mAP_mean_teacher = total_ap_mean_teacher / num_target_val_data

            print('mAP_mean_teacher: {}'.format(mAP_mean_teacher))

    def val(self, args):
        if self.load(args.pretrain_ckpt_sub_dir):
            print(" [*] Success loading")
        else:
            print(" [*] Failed loading")

        IMAGES_DIR = './datasets/FDDB/originalPics/'
        ANNOTATIONS_PATH = './datasets/FDDB/FDDB-folds/'
        RESULT_DIR = args.result_dir
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        # collect annotated images
        annotations = [s for s in os.listdir(ANNOTATIONS_PATH) if s.endswith('ellipseList.txt')]
        image_lists = [s for s in os.listdir(ANNOTATIONS_PATH) if not s.endswith('ellipseList.txt')]
        annotations = sorted(annotations)
        image_lists = sorted(image_lists)

        images_to_use = []
        for n in image_lists:
            with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
                images_to_use.extend(f.readlines())

        images_to_use = [s.strip() for s in images_to_use]
        with open(os.path.join(RESULT_DIR, 'faceList.txt'), 'w') as f:
            for p in images_to_use:
                f.write(p + '\n')

        ellipses = []
        for n in annotations:
            with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
                ellipses.extend(f.readlines())

        i = 0
        with open(os.path.join(RESULT_DIR, 'ellipseList.txt'), 'w') as f:
            for p in ellipses:
                # check image order
                if 'big/img' in p:
                    assert images_to_use[i] in p
                    i += 1
                f.write(p)

        # predict using trained detector
        predictions_list = []
        for n in tqdm(images_to_use):
            # load image
            image_array = cv2.imread(os.path.join(IMAGES_DIR, n) + '.jpg')
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            # preprocess
            h, w, _ = image_array.shape
            # image_array = cv2.resize(image_array, None, fx=3, fy=3)
            image_array = cv2.resize(image_array, (1024, 1024))
            image = image_array.astype(np.float32) / 255.0
            # now pixel values are scaled to [0, 1] range
            image = np.expand_dims(image, 0)
            # prediction
            predictions = self.sess.run(self.s_prediction_list, feed_dict={self.s_images: image})
            # extract prediction
            num_boxes = predictions[0]['num_boxes'][0]
            boxes = predictions[0]['boxes'][0][:num_boxes]
            scores = predictions[0]['scores'][0][:num_boxes]

            to_keep = scores > 0.05
            boxes = boxes[to_keep]
            scores = scores[to_keep]

            scaler = np.array([h, w, h, w], dtype='float32')
            boxes = boxes * scaler

            predictions_list.append((n, boxes, scores))

        # output results
        with open(os.path.join(RESULT_DIR, 'detections.txt'), 'w') as f:
            for n, boxes, scores in predictions_list:
                f.write(n + '\n')
                f.write(str(len(boxes)) + '\n')
                for b, s in zip(boxes, scores):
                    ymin, xmin, ymax, xmax = b
                    h, w = int(ymax - ymin), int(xmax - xmin)
                    f.write('{0} {1} {2} {3} {4:.4f}\n'.format(int(xmin), int(ymin), w, h, s))

        # copy images (need to evaluate)
        for n in tqdm(images_to_use):
            p = os.path.join(RESULT_DIR, 'images', n + '.jpg')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy(os.path.join(IMAGES_DIR, n) + '.jpg', p)

    def pretrain_val(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # pretraining model loading
        if self.load(is_pretrain=True):
            print(" [*] Load SUCCESS")
        else:
            assert False, " [!] Load failed..."

        # source val input_pipeline
        source_val_next_el = self.source_input_fn(is_training=False)
        # target_val_next_el = self.target_input_fn(is_training=False)
        # evaluator
        evaluator = EvaluatorNumpy()

        # ----- source -----
        try:
            source_result_dir = os.path.join(args.result_dir, args.sub_dir, 'source_on_pretrain')
            source_GT_result_dir = os.path.join(args.result_dir, args.sub_dir, 'source_GT_on_pretrain')
            makedirs(source_result_dir)
            makedirs(source_GT_result_dir)
            while tqdm(True):
                images, boxes, num_boxes, filenames = self.sess.run(source_val_next_el)
                feed_dict = {
                    self.s_images: images,
                    self.s_boxes: boxes,
                    self.s_num_boxes: num_boxes,
                }

                prediction_list = self.sess.run(self.s_prediction_list, feed_dict=feed_dict)
                output_prediction(prediction_list, boxes, num_boxes, filenames,
                                  source_result_dir, source_GT_result_dir)
        except Exception as E:
            print('finished validating source images')
            print('exception: {}'.format(E))

        # # ----- target -----
        # try:
        #     target_result_dir = os.path.join(args.result_dir, args.sub_dir, 'target_on_pretrain')
        #     target_GT_result_dir = os.path.join(args.result_dir, args.sub_dir, 'target_GT_on_pretrain')
        #     makedirs(target_result_dir)
        #     makedirs(target_GT_result_dir)
        #     while tqdm(True):
        #         images, boxes, num_boxes, filenames = self.sess.run(target_val_next_el)
        #         feed_dict = {
        #             self.s_images: images,
        #             self.s_boxes: boxes,
        #             self.s_num_boxes: num_boxes,
        #         }
        #
        #         prediction_list = self.sess.run(self.s_prediction_list, feed_dict=feed_dict)
        #         output_prediction(prediction_list, boxes, num_boxes, filenames,
        #                           target_result_dir, target_GT_result_dir)
        # except Exception as E:
        #     print('finished validating target images')
        #     print('exception: {}'.format(E))

    def get_feature_maps(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # pretraining model loading
        if self.load(is_pretrain=True):
            print(" [*] Load SUCCESS")
        else:
            assert False, " [!] Load failed..."

        # source val input_pipeline
        print('loading dataset ...')
        source_val_next_el = self.source_input_fn(is_training=False)

        # ----- source -----
        try:
            source_result_dir = os.path.join(args.result_dir, args.sub_dir, 'feature_npy')
            makedirs(source_result_dir)

            print('getting feature maps ...')
            cnt = 0
            print('')
            while True:
                images, boxes, num_boxes, filenames = self.sess.run(source_val_next_el)
                feed_dict = {
                    self.s_images: images,
                    self.s_boxes: boxes,
                    self.s_num_boxes: num_boxes,
                }

                feature_maps = self.sess.run(self.s_detector.get_feature_maps(), feed_dict=feed_dict)
                output_feature_maps(feature_maps, filenames, source_result_dir)
                cnt += 1
                print('\r%d' % cnt, end='')
        except Exception as E:
            print('\nfinished saving feature_maps as npy file.')
            print('exception: {}'.format(E))

    def save(self, step, is_pretrain=False, model_name=None):
        if model_name is None:
            model_name = "FaceBoxes_with_MT.model"

        checkpoint_dir = self.pretrain_ckpt_dir if is_pretrain else self.mt_ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, is_pretrain=False, model_name=None):
        print(" [*] Reading checkpoint...")
        if model_name is None:
            model_name = "FaceBoxes_with_MT.model"

        if is_pretrain:
            ckpt_dir = self.pretrain_ckpt_dir
        else:
            ckpt_dir = self.mt_ckpt_dir

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def source_input_fn(self, is_training=True):
        # image_size = self.image_size if is_training else None
        image_size = self.image_size
        # (for evaluation i use images of different sizes)

        # dataset_path = os.path.join(self.args.dataset_dir, self.args.source_dir, 'train') \
        #     if is_training else os.path.join(self.args.dataset_dir, self.args.source_dir, 'val')
        dataset_path = os.path.join(self.args.dataset_dir, self.args.source_dir, 'val', 'shards')

        batch_size = self.batch_size
        # for evaluation it's important to set batch_size to 1

        filenames = os.listdir(dataset_path)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            el = pipeline.get_batch()
        return el

    def target_input_fn(self, is_training=True):
        image_size = self.image_size if is_training else None
        # (for evaluation i use images of different sizes)

        dataset_path = os.path.join(self.args.dataset_dir, self.args.target_dir, 'train') \
            if is_training else os.path.join(self.args.dataset_dir, self.args.target_dir, 'val')

        batch_size = self.batch_size

        filenames = os.listdir(dataset_path)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            if is_training:
                pipeline = MeanTeacherPipeline(
                    filenames,
                    batch_size=batch_size, image_size=image_size,
                    repeat=is_training, shuffle=is_training,
                    augmentation=is_training
                )
                el = pipeline.get_batch()
            else:
                pipeline = Pipeline(
                    filenames,
                    batch_size=batch_size, image_size=image_size,
                    repeat=is_training, shuffle=is_training,
                    augmentation=is_training
                )
                el = pipeline.get_batch()
        return el


def make_cosine_similarity_matrix(feature):
    """
    Argument:
        feature: a tensor with shape (batch_size, num_box, feature_dim)
    Return:
        cosine_similarity: a tensor with shape (batch_size, num_box, num_box)
    """

    def fn(x):
        norm_A_expand = tf.expand_dims(x, axis=1)
        norm_B_expand = tf.expand_dims(x, axis=0)
        cs_matrix = tf.reduce_sum(norm_A_expand * norm_B_expand, axis=-1)
        return cs_matrix

    normalized_feature = tf.nn.l2_normalize(feature, axis=2)

    cosine_similarity = tf.map_fn(
        fn, normalized_feature, dtype=tf.float32,
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=True, swap_memory=False, infer_shape=True
    )
    return cosine_similarity


def add_weight_decay(weight_decay, scope=None):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable_vars = tf.trainable_variables()
    if scope is not None:
        assert isinstance(scope, str)
        kernels = [v for v in trainable_vars if ('weights' in v.name) and (scope in v.name)]
    else:
        kernels = [v for v in trainable_vars if 'weights' in v.name]

    sum_weights = []
    for K in kernels:
        sum_weights.append(tf.multiply(weight_decay, tf.nn.l2_loss(K)))
    return tf.reduce_sum(sum_weights)
