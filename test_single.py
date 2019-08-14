"""
Usage:
    python3 test.py <DATASET> <MODALITY> <TEST_SPLIT_LIST> <CHECKPOINTS_FILE> \
        --arch <ARCH> --save_scores <SAVE_SCORES>
"""

import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule


def options():
    # options`
    parser = argparse.ArgumentParser(
        description="Standard video-level testing")
    parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'saag01'])
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'ARP'])
    parser.add_argument('video_dir', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--save_scores', type=str, default=None)
    parser.add_argument('--num_segments', type=int, default=3)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--consensus_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--img_prefix', type=str, default='img')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--flow_prefix', type=str, default='flow')
    parser.add_argument('--custom_prefix', type=str, default='')
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=5)

    return parser


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _count_num_frames(video_dir):
    return len(video_dir)


def get_frame(video_dir, idx, modality, image_tmpl):
    if modality == 'RGB' or modality == 'RGBDiff' or modality== 'ARP':
        return [Image.open(os.path.join(directory, image_tmpl.format(idx))).convert('RGB')]
    elif modality == 'Flow':
        if idx != 1: # WHAT IS THIS FOR???
            idx = idx - 1
        x_img = Image.open(os.path.join(directory, image_tmpl.format('x', idx))).convert('L')
        y_img = Image.open(os.path.join(directory, image_tmpl.format('y', idx))).convert('L')
        return [x_img, y_img]

def _get_test_indices(num_frames, new_length, num_segments):
    tick = (num_frames - new_length + 1) / float(num_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    return offsets + 1

def video_loader(video_dir, new_length, num_segments):
    num_frames = _count_num_frames(video_dir)
    segment_indices = _get_test_indices(num_frames, new_length, num_segments)
    images = list()
    for seg_ind in segment_indices:
        p = int(seg_ind)
        for i in range(new_length):
            seg_imgs = get_frame(video_dir, p)
            images.extend(seg_imgs)
            if p < record.num_frames:
                p += 1

    return images

def get_transforms():
    torchvision.transforms.Compose([
        GroupScale(int(224)),
        GroupCenterCrop(224),
        Stack(roll=args.arch == 'BNInception'),
        ToTorchFormatTensor(div=args.arch != 'BNInception'),
        GroupNormalize(input_mean, input_std),
    ])


if __name__ == '__main__':
    parser = options()
    args = parser.parse_args()

    if args.modality in ['RGB', 'ARP']:
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'saag01':
        num_class = 2
    else:
        raise ValueError('Unknown dataset '+args.dataset)


    images = video_loader(args.video_dir, data_length, args.num_segments)
    transforms = get_transforms()
    processed_data = transforms(images)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=0.5,
                partial_bn=False)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_size = model.input_size
    input_std = model.input_std
    policies = model.get_optim_policies()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    processed_data = processed_data.cuda(async=True)
    output = model(processed_data)
    _, pred = output.topk(maxk, 1, True, True)

    print("TEST SINGLE ------------------------------------------")
    print("Outputs Prediction Class: ", pred)


