import argparse

import tensorflow as tf

from model import Model

parser = argparse.ArgumentParser(description='')

# main setting
parser.add_argument('--phase', dest='phase', default='train', help='train, test or pretrain')
parser.add_argument('--gpu_num', dest='gpu_num', type=int, default=1, help='the number of gpu you use')

# test setting
parser.add_argument('--source_only_test', dest='source_only_test', type=bool, default=False, help='pretrain test')
parser.add_argument('--MT_test', dest='MT_test', type=bool, default=False, help='mean teacher test')

# output dir setting
parser.add_argument('--sub_dir', dest='sub_dir', help='sub directory name')
parser.add_argument('--pretrain_ckpt_sub_dir', dest='pretrain_ckpt_sub_dir', type=str,
                    default='face-pretrain', help='pretrain sub dir name')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./checkpoint', type=str, help='ckpt_dir name')
parser.add_argument('--log_dir', dest='log_dir', default='./log', type=str, help='log_dir name')
parser.add_argument('--result_dir', dest='result_dir', default='./result', type=str, help='result_dir name')

# dataset dir setting
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./datasets', type=str, help='dataset dir')
parser.add_argument('--source_dir', dest='source_dir', default='face', type=str, help='source dataset dir')
parser.add_argument('--target_dir', dest='target_dir', default='ball', type=str, help='target dataset dir')

# <dataset tree>
# [root_dir]/datasets
#             |-- [source_dir]/
#             |       |-- train/
#             |       |-- val/
#             | -- [target_dir]/
#             |       | -- train/
#             |       | -- val/

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
        if args.phase == 'pretrain':
            model.pretrain()
        elif args.phase == 'train':
            model.train()
        elif args.phase == 'test':
            assert args.source_only_test or args.MT_test, "you should set True on either source_only_test or MT_test."
            model.test(args)


if __name__ == '__main__':
    tf.app.run()
