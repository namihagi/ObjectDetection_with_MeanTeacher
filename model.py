import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src import FeatureExtractor, AnchorGenerator, Detector, makedirs
from src.constants import PARALLEL_ITERATIONS
from src.evaluation_numpy import EvaluatorNumpy
from src.input_pipeline import Pipeline
from src.result_util import output_prediction, output_feature_maps

CONFIG_PATH = 'config.json'


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.is_training = (args.phase == 'train')
        self.gpu_num = args.gpu_num

        # directory setting
        self.dataset_path = args.dataset_path
        self.sub_dir = args.sub_dir
        assert self.sub_dir is not None
        self.result_dir = os.path.join('./result', args.sub_dir)
        self.log_dir = os.path.join('./log', args.sub_dir)
        self.checkpoint_dir = os.path.join('./checkpoint', args.sub_dir)
        for dir_path in [self.result_dir, self.log_dir, self.checkpoint_dir]:
            makedirs(path=dir_path)

        # input parameters
        self.batch_size = args.batch_size
        assert self.batch_size % self.gpu_num == 0
        self.image_size = args.image_size

        # parameters for model
        self.epoch = args.epoch
        assert self.epoch > 250
        dataset_size = len(os.listdir(self.dataset_path))
        self.max_iter_per_epoch = dataset_size // self.batch_size
        self.lr_boundaries = [200 * self.max_iter_per_epoch, 250 * self.max_iter_per_epoch],
        self.lr_values = [1e-3, 1e-4, 1e-5],
        self.weight_decay = 5e-4,
        self.l_loss_weight = 0.0
        self.c_loss_weight = 1.0
        self.ohem_params = {
            "loss_to_use": "classification",
            "loc_loss_weight": 0.0, "cls_loss_weight": 1.0,
            "num_hard_examples": 500, "nms_threshold": 0.99,
            "max_negatives_per_positive": 3.0, "min_negatives_per_image": 30,
        }
        # parameters for predictions
        self.score_threshold = 0.3,
        self.iou_threshold = 0.3,
        self.max_boxes = 200,

        # build graph
        self._build_model()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # create writer and saver
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()

    def _build_model(self):
        # input placeholder
        self.images = tf.placeholder(
            tf.float32,
            [self.batch_size, self.image_size, self.image_size, 3],
            name="images"
        )
        self.boxes = tf.placeholder(tf.float32, [self.batch_size, None, 4], name="boxes")
        self.num_boxes = tf.placeholder(tf.int32, [self.batch_size], name="num_boxes")

        # input split
        images_per_gpu = tf.split(self.images, self.gpu_num)
        boxes_per_gpu = tf.split(self.images, self.gpu_num)
        num_boxes_per_gpu = tf.split(self.images, self.gpu_num)

        # each gpu model define
        for gpu_id in range(int(self.gpu_num)):
            reuse = (gpu_id > 0)
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope('detector', reuse=reuse):
                    feature_extractor = FeatureExtractor(is_training=self.is_training)
                    anchor_generator = AnchorGenerator()
                    detector = Detector(images_per_gpu[gpu_id], feature_extractor, anchor_generator)

                    with tf.name_scope('weight_decay'):
                        r_loss = add_weight_decay(self.weight_decay, scope='detector')
                        tf.add_to_collection('r_loss', r_loss)

                    with tf.name_scope('loss'):
                        labels = {'boxes': boxes_per_gpu[gpu_id], 'num_boxes': num_boxes_per_gpu[gpu_id]}
                        losses = detector.loss(labels, self.ohem_params)
                        l_loss = self.l_loss_weight * losses['localization_loss']
                        c_loss = self.c_loss_weight * losses['classification_loss']
                        total_loss_per_gpu = l_loss + c_loss + r_loss
                        tf.add_to_collection('total_loss_per_gpu', total_loss_per_gpu)
                        tf.add_to_collection('l_loss', l_loss)
                        tf.add_to_collection('c_loss', c_loss)

                    with tf.name_scope('prediction'):
                        prediction = detector.get_predictions(
                            score_threshold=self.score_threshold,
                            iou_threshold=self.iou_threshold,
                            max_boxes=self.max_boxes
                        )
                        tf.add_to_collection('prediction', prediction)

        self.total_loss = tf.reduce_mean(tf.get_collection('total_loss_per_gpu', scope='detector'))
        self.total_r_loss = tf.reduce_mean(tf.get_collection('r_loss', scope='detector'))
        self.total_l_loss = tf.reduce_mean(tf.get_collection('l_loss', scope='detector'))
        self.total_c_loss = tf.reduce_mean(tf.get_collection('c_loss', scope='detector'))
        self.total_loss = tf.get_collection('prediction', scope='detector')

        # learning rate
        with tf.variable_scope('learning_rate'):
            self.global_step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.piecewise_constant(self.global_step, self.lr_boundaries, self.lr_values)

        # getting variables
        self.t_vars = tf.trainable_variables(scope='detector')
        print('----- detector variables -----')
        for var in self.t_vars:
            print(var.name)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='detector')
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)
            self.train_op = optimizer.minimize(
                self.total_loss, global_step=self.global_step,
                colocate_gradients_with_ops=(self.gpu_num > 1), var_list=self.t_vars
            )

        # summary
        regularization_loss_sum = tf.summary.scalar('regularization_loss', self.total_r_loss)
        s_localization_loss_sum = tf.summary.scalar('localization_loss', self.total_l_loss)
        s_classification_loss_sum = tf.summary.scalar('classification_loss', self.total_c_loss)
        s_supervised_total_loss_sum = tf.summary.scalar('total_loss', self.total_loss)
        s_lr_sum = tf.summary.scalar('learning_rate', self.learning_rate)
        self.merged_source_sum = tf.summary.merge(
            [s_lr_sum, s_supervised_total_loss_sum, regularization_loss_sum,
             s_localization_loss_sum, s_classification_loss_sum]
        )

    def train(self):
        # training input
        init_op, next_el, num_files = self.get_input_op()

        print('start training')
        for idx_epoch in tqdm(range(self.epoch)):
            self.sess.run(init_op)
            for idx_iter in tqdm(range(self.max_iter_per_epoch), leave=False):
                images, boxes, num_boxes, filenames = self.sess.run(next_el)
                feed_dict = {
                    self.images: images,
                    self.boxes: boxes,
                    self.num_boxes: num_boxes,
                }
                global_step, _, summary = self.sess.run(
                    [self.global_step, self.train_op, self.merged_source_sum],
                    feed_dict=feed_dict
                )
                self.writer.add_summary(summary, int(global_step))

        # save model
        self.save()

    def test(self, args):
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
            predictions = self.sess.run(self.prediction_list, feed_dict={self.images: image})
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
                    self.images: images,
                    self.boxes: boxes,
                    self.num_boxes: num_boxes,
                }

                prediction_list = self.sess.run(self.prediction_list, feed_dict=feed_dict)
                output_prediction(prediction_list, boxes, num_boxes, filenames,
                                  source_result_dir, source_GT_result_dir)
        except Exception as E:
            print('finished validating source images')
            print('exception: {}'.format(E))

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
                    self.images: images,
                    self.boxes: boxes,
                    self.num_boxes: num_boxes,
                }

                feature_maps = self.sess.run(self.detector.get_feature_maps(), feed_dict=feed_dict)
                output_feature_maps(feature_maps, filenames, source_result_dir)
                cnt += 1
                print('\r%d' % cnt, end='')
        except Exception as E:
            print('\nfinished saving feature_maps as npy file.')
            print('exception: {}'.format(E))

    def save(self, model_name=None):
        if model_name is None:
            model_name = "FaceBoxes.model"

        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))
        print(' [*] saved model')

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

    def get_input_op(self):
        filenames = os.listdir(self.dataset_path)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(self.dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames, batch_size=self.batch_size, image_size=self.image_size,
                shuffle=self.is_training, augmentation=self.is_training
            )
        return pipeline.get_init_op_and_next_el()


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
