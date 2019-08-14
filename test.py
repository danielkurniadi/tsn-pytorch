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
    parser.add_argument('test_list', type=str)
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


def test(model, test_loader, args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    Logits = []
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(input_var)

        print(output.size())

        # precision
        prec1, prec5 = accuracy(output.data, target, topk=(1,2))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time,
                   top1=top1, top5=top5)))

        Logits.append(output.cpu().detach().numpy())
    
    stacked = np.concatenate(Logits)
    np.save("scores/saag01_bni_flow_seg_3_test_scores.npy", stacked)

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
          .format(top1=top1, top5=top5)))


def main():
    parser = options()
    args = parser.parse_args()

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

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

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
    train_augmentation = model.get_augmentation()

    cropping = torchvision.transforms.Compose([
        GroupScale(scale_size),
        GroupCenterCrop(input_size),
    ])

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    
    state_dict = checkpoint['state_dict']

    # base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(state_dict)

    test_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list,
            num_segments=args.num_segments,
            new_length=data_length,
            modality=args.modality,
            image_tmpl=args.img_prefix + "_{:05d}" + args.ext if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"_{}_{:05d}" + args.ext,
            random_shift=False,
            transform=torchvision.transforms.Compose([
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=args.arch == 'BNInception'),
                ToTorchFormatTensor(div=args.arch != 'BNInception'),
                GroupNormalize(input_mean, input_std),
            ]),
            custom_prefix = args.custom_prefix
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True
    )

    ### Test ###
    test(model, test_loader, args)


if __name__ == '__main__':
    main()

