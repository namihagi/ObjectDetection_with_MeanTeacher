import argparse

import tensorflow as tf

from model import Model

parser = argparse.ArgumentParser(description='')

# main setting
parser.add_argument('--phase', dest='phase', default='train', help='train, test, pretrain_val or pretrain')
parser.add_argument('--gpu_num', dest='gpu_num', type=int, default=1, help='the number of gpu you use')

# model setting
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='the number of images in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='how many times you repeat dataset')
parser.add_argument('--image_size', dest='image_size', type=int, default=1024, help='load size of image')
parser.add_argument('--class_name', dest='class_name', type=str, default='face', help='input class name')

# parameters for prediction
parser.add_argument('--score_threshold', dest='score_threshold', type=float,
                    default=0.05, help='get prediction whose score is larger than this')
parser.add_argument('--iou_threshold', dest='iou_threshold', type=float, default=0.3)
parser.add_argument('--max_boxes', dest='max_boxes', type=int, default=200, help='max predictions per a image')

# dir setting
parser.add_argument('--dataset_path', dest='dataset_path', type=str,
                    help='path to dataset shards directory or a image')
parser.add_argument('--sub_dir', dest='sub_dir', type=str, help='sub directory name')
parser.add_argument('--output_dir', dest='output_dir', type=str, default=None,
                    help='output directory for predictions')
parser.add_argument('--input_image_dir', dest='input_image_dir', type=str, default=None,
                    help='input directory for detection by feature')

args = parser.parse_args()


def main(_):
    # graph config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # gpu_num config
    usable_gpu = "0"
    for idx in range(args.gpu_num - 1):
        usable_gpu += ",{}".format(idx + 1)
    tfconfig.gpu_options.visible_device_list = usable_gpu

    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        if args.phase == 'train':
            model.train()
        elif args.phase == 'test_for_FDDB':
            model.test_for_FDDB()
        elif args.phase == 'test_for_VOC':
            model.test_for_VOC(args)
        elif args.phase == 'get_prediction_from_images':
            model.get_prediction_from_images(args)
        elif args.phase == 'get_feature_maps':
            model.get_feature_maps(args)
        elif args.phase == 'input_feature_detection':
            model.get_prediction_from_features(args)


if __name__ == '__main__':
    tf.app.run()
