import argparse

import tensorflow as tf

from model import Model

parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='train', help='train, test or pretrain')
parser.add_argument('--sub_dir', dest='sub_dir', help='sub directory name')
parser.add_argument('--gpu_num', dest='gpu_num', type=int, default=1, help='the number of gpu you use')
parser.add_argument('--pretrain_ckpt_dir', dest='pretrain_ckpt_dir', default='face-pretrain', help='pretrain directory name')

parser.add_argument('--source_only_test', dest='source_only_test', type=bool, default=False, help='pretrain test')
parser.add_argument('--MT_test', dest='MT_test', type=bool, default=False, help='mean teacher test')

args = parser.parse_args()


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
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
