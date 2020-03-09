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

# dir setting
parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='path to dataset shards directory')
parser.add_argument('--sub_dir', dest='sub_dir', type=str, help='sub directory name')

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
        elif args.phase == 'test':
            model.test()

        # if args.phase == 'pretrain':
        #     model.pretrain()
        # elif args.phase == 'train':
        #     model.train()
        # elif args.phase == 'pretrain_val':
        #     model.pretrain_val(args)
        # elif args.phase == 'test':
        #     assert args.source_only_test or args.MT_test, "you should set True on either source_only_test or MT_test."
        #     model.test(args)
        # elif args.phase == 'get_feature_maps':
        #     model.get_feature_maps(args)
        # elif args.phase == 'val':
        #     model.val(args)


if __name__ == '__main__':
    tf.app.run()
