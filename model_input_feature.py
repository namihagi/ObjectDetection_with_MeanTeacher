import json
import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from src import AnchorGenerator, makedirs
from src.detector_input_feature import Detector
from src.network_input_feature import FeatureExtractor
from src.result_util import output_prediction_from_image

CONFIG_PATH = 'config.json'

half = 300
d_min = int(1024 / 2) - 300
d_max = int(1024 / 2) + 300
DAMY_BOX = [d_min / 1024, d_min / 1024, d_max / 1024, d_max / 1024]


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        # load parameters
        self.params = json.load(open(CONFIG_PATH))
        self.model_params = self.params['model_params']
        self.input_params = self.params['input_pipeline_params']

        # model parameters
        self.norm_phase = args.norm_phase

        # input parameters
        self.batch_size = args.batch_size
        self.input_npy_dir = args.input_npy_dir
        self.input_image_dir = args.input_image_dir
        self.class_name = args.class_name
        self.output_dir = args.output_dir
        makedirs(self.output_dir)
        self.output_filename = args.output_filename

        # checkpoint dir
        self.ckpt_dir = args.ckpt_dir

        # build graph
        self._build_model()

        # create writer and saver
        self.saver = tf.train.Saver()

        # initialize weights
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # pretraining model loading
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            assert False, " [!] Load failed..."

    def _build_model(self):
        # source input placeholder
        self.s_images = tf.placeholder(tf.float32, [None, None, None, 3], name="s_images")
        self.s_features = tf.placeholder(tf.float32, [None, 32, 32, 128], name="s_features")

        # student model
        with tf.variable_scope('student'):
            self.s_feature_extractor = FeatureExtractor(is_training=False)
            self.s_anchor_generator = AnchorGenerator()
            self.s_detector = Detector(self.s_features, self.s_images,
                                       self.s_feature_extractor, self.s_anchor_generator)

            with tf.name_scope('student_prediction'):
                self.s_prediction = self.s_detector.get_predictions(
                    score_threshold=self.model_params['score_threshold'],
                    iou_threshold=self.model_params['iou_threshold'],
                    max_boxes=self.model_params['max_boxes']
                )

        # getting variables
        self.student_vars = tf.trainable_variables(scope='student')
        print('----- student model variables -----')
        for var in self.student_vars:
            print(var.name)

    def get_predictions_random(self):
        # get data
        dataset_files = self.get_input_fn()

        pred_box_num_list = {}
        pred_box_sum_list = {}
        large_area_list = {}
        for dataset_file in dataset_files:
            # load npy
            feature_array = np.load(dataset_file).astype(np.float32)
            damy_images = np.zeros((self.batch_size, 1024, 1024, 3), dtype=np.float32)

            batch_iter = 10000 // self.batch_size

            box_num = 0
            box_num_dist = {}
            large_area_num = 0
            with tqdm(range(batch_iter)) as bar:
                for idx in bar:
                    bar.set_description('{}, box_num {}'.format(dataset_file.split('/')[-1], box_num))

                    # translate feature
                    feature = feature_array[self.batch_size * idx:self.batch_size * (idx + 1)]
                    feature = self.translation(feature)

                    # prediction
                    predictions = self.sess.run(self.s_prediction,
                                                feed_dict={self.s_images: damy_images,
                                                           self.s_features: feature})

                    # validation for each prediction
                    for prediction in predictions:
                        for pred_scores, pred_boxes in zip(prediction['scores'], prediction['boxes']):
                            pred_scores = np.array(pred_scores, dtype=np.float32)

                            # extract true prediction
                            large_score = pred_scores[pred_scores > 0.5]
                            large_box = pred_boxes[pred_scores > 0.5]

                            # count pred_boxes in a feature
                            box_num += large_score.size
                            if int(large_score.size) in box_num_dist.keys():
                                box_num_dist[int(large_score.size)] += 1
                            else:
                                box_num_dist[int(large_score.size)] = 1

                            # check whether Bounding Box area is larger than face
                            for box in large_box:
                                if IoU(box, DAMY_BOX) > 0.5:
                                    large_area_num += 1

            pred_box_num_list['{}'.format(dataset_file.split('/')[-1])] = box_num
            pred_box_sum_list['{}'.format(dataset_file.split('/')[-1])] = box_num_dist
            large_area_list['{}'.format(dataset_file.split('/')[-1])] = large_area_num

        # output sum of boxes in each file
        with open(os.path.join(self.output_dir, self.output_filename), 'w') as fout:
            pred_box_num_list = sorted(pred_box_num_list.items())
            json.dump(pred_box_num_list, fout)

        # output distribution of box num
        with open(os.path.join(self.output_dir, self.output_filename.replace('_log', '_distribution')), 'w') as fout:
            json.dump(pred_box_sum_list, fout)

        # output box num which has large area
        with open(os.path.join(self.output_dir, self.output_filename.replace('_log', '_area')), 'w') as fout:
            json.dump(large_area_list, fout)

    def get_predictions_by_cyclegan(self):
        assert self.batch_size == 1, 'global batch size need to be 1. [now] {}'.format(self.batch_size)

        # get npy files
        dataset_file_dict = self.get_npy_fn_for_image()

        # make dir for each checkpoint result.
        output_dir = os.path.join(self.output_dir)
        makedirs(output_dir)

        with tqdm(dataset_file_dict) as bar:
            for file_path in bar:
                # load npy and image
                feature = np.load(file_path).astype(np.float32)
                # self.translation(feature)
                image = self.load_image(file_path)
                assert len(image.shape) == 3
                img_w, img_h = image.shape[0:2]

                # expand dims for NHWC
                if len(feature.shape) == 3:
                    feature = np.expand_dims(feature, axis=0)
                if len(image.shape) == 3:
                    image = np.zeros((self.batch_size, 1024, 1024, 3), dtype=np.float32)
                assert len(feature.shape) == 4 and len(image.shape) == 4

                # prediction
                prediction = self.sess.run(self.s_prediction,
                                           feed_dict={self.s_images: image,
                                                      self.s_features: feature})
                pred_scores = np.array(prediction['scores'], dtype=np.float32)
                pred_boxes = prediction['boxes']

                # extract true prediction
                pred_boxes = pred_boxes[pred_scores > 0.5]
                pred_scores = pred_scores[pred_scores > 0.5]
                assert len(pred_scores) == len(pred_boxes)

                output_prediction_from_image(pred_scores, pred_boxes, img_w, img_h,
                                             self.class_name, output_dir, file_path)

    def get_predictions_image(self):
        assert self.batch_size == 1, 'global batch size need to be 1. [now] {}'.format(self.batch_size)

        # get npy files
        dataset_file_dict = self.get_npy_fn_for_image()

        for checkpoint, file_paths in tqdm(dataset_file_dict.items()):
            # make dir for each checkpoint result.
            output_dir = os.path.join(self.output_dir, checkpoint)
            makedirs(output_dir)

            with tqdm(file_paths, leave=False) as bar:
                for file_path in bar:
                    bar.set_description('{}'.format(checkpoint))

                    # load npy and image
                    feature = np.load(file_path).astype(np.float32)
                    feature = np.clip(feature, 0.0, 14.8)
                    image = self.load_image(file_path)

                    # expand dims for NHWC
                    feature = np.expand_dims(feature, axis=0)
                    image = np.expand_dims(image, axis=0)
                    img_w, img_h = image.shape[1:3]
                    assert len(feature.shape) == 4 and len(image.shape) == 4

                    # prediction
                    prediction = self.sess.run(self.s_prediction,
                                               feed_dict={self.s_images: image,
                                                          self.s_features: feature})
                    pred_scores = np.array(prediction['scores'], dtype=np.float32)
                    pred_boxes = prediction['boxes']

                    # extract true prediction
                    pred_boxes = pred_boxes[pred_scores > 0.5]
                    pred_scores = pred_scores[pred_scores > 0.5]
                    assert len(pred_scores) == len(pred_boxes)

                    output_prediction_from_image(pred_scores, pred_boxes, img_w, img_h,
                                                 self.class_name, output_dir, file_path)

        # # draw box
        # img = Image.open('./datasets/ffhq/images/00000.png')
        # draw = ImageDraw.Draw(img)
        #
        # for score, box in zip(pred_score, pred_box):
        #     draw.rectangle(((box[1], box[0]), (box[3], box[2])), outline=(255, 0, 0))
        #     print(score)
        #     print('xmin: %f, xmax: %f, ymin: %f, ymax: %f' % (box[1], box[3], box[0], box[2]))
        #
        # img.save('./test.png')

    def translation(self, feature):
        if self.norm_phase == 'normalized':
            feature = feature * 0.5232219099998474 + 0.2492203712463379
        elif self.norm_phase == 'log':
            feature = np.exp(feature) - 1

        return feature

    def get_input_fn(self):
        files = sorted(glob(os.path.join(self.input_npy_dir, '0*.npy')))
        files += [os.path.join(self.input_npy_dir, 'final.npy')]
        return files

    def get_npy_fn_for_image(self, add_dir=None):
        # # get file_paths in all checkpoints dir.
        # checkpoint_list = sorted(os.listdir(self.input_npy_dir))
        # filename_list = {}
        # for checkpoint in checkpoint_list:
        #     if add_dir is not None:
        #         files = sorted(glob(os.path.join(self.input_npy_dir, checkpoint, add_dir, '*.npy')))
        #     else:
        #         files = sorted(glob(os.path.join(self.input_npy_dir, checkpoint, '*.npy')))
        #     if len(files) == 0:
        #         continue
        #     filename_list[checkpoint] = files
        filename_list = sorted(glob(os.path.join(self.input_npy_dir, '*.npy')))
        return filename_list

    def load_image(self, file_path):
        image_name = file_path.split('/')[-1].replace('.npy', '.jpg')
        image = cv2.imread(os.path.join(self.input_image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).astype(np.float32)
        return image

    def load(self):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


def IoU(box1, box2):
    ymin1 = box1[0] * 1024
    xmin1 = box1[1] * 1024
    ymax1 = box1[2] * 1024
    xmax1 = box1[3] * 1024

    ymin2 = box2[0] * 1024
    xmin2 = box2[1] * 1024
    ymax2 = box2[2] * 1024
    xmax2 = box2[3] * 1024

    over_xmin = max(xmin1, xmin2)
    over_ymin = max(ymin1, ymin2)
    over_xmax = min(xmax1, xmax2)
    over_ymax = min(ymax1, ymax2)

    box1_area = (ymax1 - ymin1) * (xmax1 - xmin1)
    box2_area = (ymax2 - ymin2) * (xmax2 - xmin2)
    over_area = (over_ymax - over_ymin) * (over_xmax - over_xmin)

    return over_area / (box1_area + box2_area - over_area)
