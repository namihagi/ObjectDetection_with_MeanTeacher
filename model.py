import json
import os

import tensorflow as tf
from tqdm import tqdm

from src import FeatureExtractor, AnchorGenerator, Detector, GraphExtractor
from src.constants import PARALLEL_ITERATIONS
from src.evaluation_numpy import EvaluatorNumpy
from src.input_pipeline import Pipeline
from src.input_pipeline.mean_teacher_pipeline import MeanTeacherPipeline

CONFIG_PATH = 'config.json'


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        # load parameters
        self.params = json.load(open(CONFIG_PATH))
        self.model_params = self.params['model_params']
        self.input_params = self.params['input_pipeline_params']
        self.dir_params = self.params['directory_params']

        # FaceBoxes model parameters
        self.epoch_source_only = self.model_params['epoch_source_only']
        self.localization_loss_weight = self.model_params['localization_loss_weight']
        self.classification_loss_weight = self.model_params['classification_loss_weight']
        self.weight_decay = self.model_params['weight_decay']

        # mean teacher parameters
        self.epoch_mean_teacher = self.model_params['epoch_mean_teacher']
        self.mean_teacher_lr = self.model_params['mean_teacher_lr']
        self.mt_lambda = self.model_params['mt_lambda']
        self.score_threshold = self.model_params['confidence_threshold']
        self.alpha = self.model_params['smooth_param_of_ema']

        # input parameters
        self.gpu_num = args.gpu_num
        if args.phase == 'pretrain':
            self.batch_size = self.input_params['pretrain_batch_size'] * self.gpu_num
        else:
            self.batch_size = self.input_params['mt_batch_size'] * self.gpu_num
        self.image_size = self.input_params['image_size']

        # directory setting
        if args.phase == 'pretrain':
            self.log_dir = os.path.join(self.dir_params['log_dir'], args.sub_dir, 'pretrain')
        else:
            self.log_dir = os.path.join(self.dir_params['log_dir'], args.sub_dir, 'train')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.pretrain_ckpt_dir = os.path.join(self.dir_params['checkpoint_dir'], args.pretrain_ckpt_dir)
        # self.pretrain_ckpt_dir = os.path.join(self.dir_params['checkpoint_dir'], args.sub_dir, 'pretrain')
        self.mt_ckpt_dir = os.path.join(self.dir_params['checkpoint_dir'], args.sub_dir, 'mean_teacher')
        if not os.path.exists(self.pretrain_ckpt_dir):
            os.makedirs(self.pretrain_ckpt_dir)
        if not os.path.exists(self.mt_ckpt_dir):
            os.makedirs(self.mt_ckpt_dir)

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
        self.s_supervised_total_loss_list = []
        self.losses_cons = []

        # list to gather predictions
        self.s_prediction_list = []

        # each gpu model define
        for gpu_id in range(int(self.gpu_num)):
            reuse = (gpu_id > 0)
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    # student model
                    with tf.variable_scope('student'):
                        self.s_feature_extractor = FeatureExtractor(is_training=True)
                        self.s_anchor_generator = AnchorGenerator()
                        self.s_detector = Detector(self.s_images, self.s_feature_extractor, self.s_anchor_generator)

                        with tf.name_scope('weight_decay'):
                            add_weight_decay(self.weight_decay, scope='student')
                            self.regularization_loss = tf.losses.get_regularization_loss()

                        with tf.name_scope('student_supervised_loss'):
                            s_labels = {'boxes': self.s_boxes, 'num_boxes': self.s_num_boxes}
                            s_losses = self.s_detector.loss(s_labels, self.model_params)
                            self.s_localization_loss = self.localization_loss_weight * s_losses['localization_loss']
                            self.s_classification_loss = \
                                self.classification_loss_weight * s_losses['classification_loss']
                            self.s_supervised_total_loss = \
                                self.s_localization_loss + self.s_classification_loss + self.regularization_loss
                            self.s_supervised_total_loss_list.append(self.s_supervised_total_loss)

                        with tf.name_scope('student_prediction'):
                            s_prediction = self.s_detector.get_predictions(
                                score_threshold=self.model_params['score_threshold'],
                                iou_threshold=self.model_params['iou_threshold'],
                                max_boxes=self.model_params['max_boxes']
                            )
                            self.s_prediction_list.append(s_prediction)

                    # teacher model
                    with tf.variable_scope('teacher'):
                        self.t_feature_extractor = FeatureExtractor(is_training=True)
                        self.t_anchor_generator = AnchorGenerator()
                        self.t_detector = Detector(self.t_images, self.t_feature_extractor, self.t_anchor_generator)

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
                        self.loss_AGL = tf.reduce_mean(tf.reduce_mean((tf.ones_like(self.s_AM) - self.s_AM), axis=(1, 2)))

                    # total consistency loss
                    self.loss_cons = self.loss_RCL + self.loss_EGL + self.loss_AGL
                    self.losses_cons.append(self.loss_cons)

        # compute all loss over gpu
        self.s_total_loss = tf.reduce_mean(tf.stack(self.s_supervised_total_loss_list, axis=0))
        self.total_cons_loss = tf.reduce_mean(tf.stack(self.losses_cons, axis=0))

        # learning rate for supervised optimizer
        with tf.variable_scope('learning_rate'):
            self.global_step = tf.train.get_global_step()
            if self.global_step is None:
                self.global_step = tf.train.create_global_step()
            self.source_learning_rate = tf.train.piecewise_constant(self.global_step,
                                                                    self.model_params['lr_boundaries'],
                                                                    self.model_params['lr_values'])

        # getting variables
        self.student_vars = tf.trainable_variables(scope='student')
        self.teacher_vars = tf.trainable_variables(scope='teacher')
        print('----- student model variables -----')
        for var in self.student_vars:
            print(var.name)
        print('----- teacher model variables -----')
        for var in self.teacher_vars:
            print(var.name)

        # this optimizer is used with only supervised training
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
        # regularization_loss_sum = tf.summary.scalar('regularization_loss', self.regularization_loss)
        # s_localization_loss_sum = tf.summary.scalar('localization_loss', self.s_localization_loss)
        # s_classification_loss_sum = tf.summary.scalar('classification_loss', self.s_classification_loss)
        s_supervised_total_loss_sum = tf.summary.scalar('s_total_loss', self.s_total_loss)
        s_lr_sum = tf.summary.scalar('learning_rate', self.source_learning_rate)
        self.merged_source_sum = tf.summary.merge([s_supervised_total_loss_sum, s_lr_sum])

        # loss_RCL_sum = tf.summary.scalar('RCL_loss', self.loss_RCL)
        # loss_EGL_sum = tf.summary.scalar('EGL_loss', self.loss_EGL)
        # loss_AGL_sum = tf.summary.scalar('AGL_loss', self.loss_AGL)
        total_cons_loss_sum = tf.summary.scalar('total_cons_loss', self.total_cons_loss)
        self.merged_MT_sum = tf.summary.merge([s_supervised_total_loss_sum, total_cons_loss_sum])

        # teacher weights update
        self.mt_initial_op = tf.group([tf.assign(t_var, s_var)
                                       for s_var, t_var in zip(self.student_vars, self.teacher_vars)])
        self.mt_update_op = tf.group([tf.assign(t_var, self.alpha * t_var + (1. - self.alpha) * s_var)
                                      for s_var, t_var in zip(self.student_vars, self.teacher_vars)])

    def pretrain(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # source training input
        train_next_el, num_train_data = self.source_input_fn(is_training=True)
        # val_next_el, num_val_data = self.source_input_fn(is_training=False)
        # evaluator = EvaluatorNumpy()

        print('start pretraining')
        # pretrain source images
        for idx in tqdm(range(self.epoch_source_only * num_train_data)):
            images, boxes, num_boxes, filenames = self.sess.run(train_next_el)
            feed_dict = {
                self.s_images: images,
                self.s_boxes: boxes,
                self.s_num_boxes: num_boxes,
            }
            _, summary = self.sess.run([self.s_optim_with_supervision, self.merged_source_sum], feed_dict=feed_dict)
            self.writer.add_summary(summary, idx)

            # if idx + 1 % 500 == 0:
            #     self.val(evaluator, val_next_el, idx)

        # save pretrain model
        global_index = self.epoch_source_only * num_train_data
        self.save(step=global_index, is_pretrain=True, model_name='pretrain.model')
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

    def val(self, evaluator, val_next_el, step):
        # val input
        images, boxes, num_boxes, filenames = self.sess.run(val_next_el)
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

        # val summary
        AP_sum = tf.summary.scalar('AP', eval_result['AP'])
        recall_sum = tf.summary.scalar('recall', eval_result['recall'])
        FP_sum = tf.summary.scalar('FP', eval_result['FP'])
        FN_sum = tf.summary.scalar('FN', eval_result['FN'])
        mean_iou_sum = tf.summary.scalar('mean_iou', eval_result['mean_iou'])
        merged_val_sum = tf.summary.merge([
            AP_sum, recall_sum, FP_sum, FN_sum, mean_iou_sum
        ])
        global_step = tf.train.get_global_step()
        summary = self.sess.run(merged_val_sum)
        self.writer.add_summary(summary, global_step=step)

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
        image_size = self.image_size if is_training else None
        # (for evaluation i use images of different sizes)
        dataset_path = self.input_params['train_source_dataset'] \
            if is_training else self.input_params['val_source_dataset']
        batch_size = self.batch_size if is_training else 1
        # for evaluation it's important to set batch_size to 1

        filenames = os.listdir(dataset_path)
        num_files = len(filenames)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=True, shuffle=is_training,
                augmentation=is_training
            )
            el = pipeline.get_batch()
        return el, int(num_files / batch_size)

    def target_input_fn(self, is_training=True):
        image_size = self.image_size if is_training else None
        # (for evaluation i use images of different sizes)
        dataset_path = self.input_params['train_target_dataset'] \
            if is_training else self.input_params['val_target_dataset']
        batch_size = self.batch_size if is_training else 1
        # for evaluation it's important to set batch_size to 1

        filenames = os.listdir(dataset_path)
        num_files = len(filenames)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            if is_training:
                pipeline = MeanTeacherPipeline(
                    filenames,
                    batch_size=batch_size, image_size=image_size,
                    repeat=True, shuffle=is_training,
                    augmentation=is_training
                )
                el = pipeline.get_batch()
            else:
                pipeline = Pipeline(
                    filenames,
                    batch_size=batch_size, image_size=image_size,
                    repeat=True, shuffle=is_training,
                    augmentation=is_training
                )
                el = pipeline.get_batch()
        return el, int(num_files / batch_size)


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
