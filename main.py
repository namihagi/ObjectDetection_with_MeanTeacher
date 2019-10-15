import argparse

import tensorflow as tf

from model import Model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test or pretrain')
parser.add_argument('--sub_dir', dest='sub_dir', help='sub directory name')
args = parser.parse_args()


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        if args.phase == 'pretrain':
            model.pretrain()
        elif args.phase == 'train':
            model.train()
        elif args.phase == 'test':
            model.test()


if __name__ == '__main__':
    tf.app.run()
