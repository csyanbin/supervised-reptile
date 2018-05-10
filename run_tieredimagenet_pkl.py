"""
Train a model on miniImageNet.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import MiniImageNetModel
from supervised_reptile.miniimagenet import read_dataset
from supervised_reptile.train import train

from dataset_tiered import *

DATA_DIR = 'data/tieredImagenet'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    # train_set, val_set, test_set = read_dataset(DATA_DIR)
    # load pkl dataset here
    n_examples = 600
    n_episodes = 100
    args_data = {}
    args_data['x_dim'] = '84,84,3'
    args_data['ratio'] = 1.0
    args_data['seed'] = 1000
    train_set = dataset_tiered(n_examples, n_episodes, 'train', args_data)
    val_set   = dataset_tiered(n_examples, n_episodes, 'val', args_data)
    test_set  = dataset_tiered(n_examples, n_episodes, 'test', args_data)
    train_set.load_data_pkl()
    val_set.load_data_pkl()
    test_set.load_data_pkl()

    model = MiniImageNetModel(args.classes, **model_kwargs(args))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            try:
                print(tf.train.latest_checkpoint(args.checkpoint))
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            except:
                print(args.checkpoint)
                tf.train.Saver().restore(sess, args.checkpoint)


        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        #print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        #print('Validation accuracy: ' + str(evaluate(sess, model, val_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
