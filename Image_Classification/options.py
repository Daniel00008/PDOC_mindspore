import argparse
import models

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Training')
# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-dset', '--dataset', default='dataset', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')

# Optimization options
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate (default: 0.001 | for inception recommend 0.0256)')
parser.add_argument('--lr-decay', default=0.1, type=float, metavar='LD',
                    help='every lr-decay-step epochs learning rate decays by LD (default:0.1 | for inception recommend 0.16)')
parser.add_argument('--lr-mode', default='step', type=str, help='learning rate mode')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, metavar='WD', help='weight decay for sgd (default: 1e-4)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--den-target', default=0.5, type=float, help='target density of the mask.')
parser.add_argument('--lbda', default=5, type=float, help='penalty factor of the L2 loss for mask.')
parser.add_argument('--gamma', default=1, type=float, help='penalty factor of the L2 loss for balance gate.')
parser.add_argument('--alpha', default=5e-2, type=float, help='alpha in exp annealing.')

# Training
parser.add_argument('--epochs', default=70, type=int, metavar='EPOCHS', help='number of total iteration to run.')

# Miscs
parser.add_argument('--manualSeed', default=3407, type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model: ''pytorch: use pytorch official | path to self-trained model')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to store the checkpoint and log checkpoint path = ./checkpoints/PATH, log path = ./logs/PATH')
parser.add_argument('--bias', default=2, type=float, help='initial value of the bias in the last fc layer of mask module.')
parser.add_argument("--task", default='PACS', type=str)
parser.add_argument("--source", default='photo', type=str, help="Target")
parser.add_argument("--image_size", type=int, default=224, help="Image size")
parser.add_argument("--seed", type=int, default=1, help="random seed")
# data aug stuff
parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
parser.add_argument("--jitter", default=0., type=float, help="Color jitter amount")
parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")

# single domain or multiple domain
parser.add_argument("--single", dest='single', action='store_true', help='evaluate model on validation set')

