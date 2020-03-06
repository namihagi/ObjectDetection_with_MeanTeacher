import argparse

import tensorflow as tf

from model_input_feature import Model

parser = argparse.ArgumentParser(description='')

# main setting
parser.add_argument('--type', dest='type', default='image', type=str, help='batch_size for predictions')
parser.add_argument('--batch_size', dest='batch_size', default=8, type=int, help='batch_size for predictions')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', help='checkpoint dir')
parser.add_argument('--input_npy_dir', dest='input_npy_dir', help='input npy dir')
parser.add_argument('--input_image_dir', dest='input_image_dir', default=None, help='input image dir')
parser.add_argument('--class_name', dest='class_name', default='aeroplane', help='class_name')
parser.add_argument('--output_dir', dest='output_dir', help='output dir')
parser.add_argument('--output_filename', dest='output_filename', help='output filename')
parser.add_argument('--norm_phase', dest='norm_phase', default="base", help='translation')

args = parser.parse_args()


def main(_):
    # graph config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # gpu_num config
    usable_gpu = "0"
    tfconfig.gpu_options.visible_device_list = usable_gpu

    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        # model.get_predictions_random()
        if args.type == 'image':
            model.get_predictions_image()
        elif args.type == 'cyclegan':
            model.get_predictions_by_cyclegan()


if __name__ == '__main__':
    tf.app.run()
