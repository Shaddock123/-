import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model import resnet18

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Test')
parser.add_argument('--net_type', default='resnet18', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--pretrained', default='./runs/TEST/', type=str, metavar='PATH')
parser.add_argument('--cutout', type=bool, default=False,
                    help='apply cutout')
parser.add_argument('--mixup', type=bool, default=False,
                    help='apply mixup')
parser.add_argument('--cutmix', type=bool, default=False,
                    help='apply mixup')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)


def main():
    global args
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    print("=> creating model '{}'".format(args.net_type))
    model = resnet18(num_classes=100)
    model = model.cuda()

    if args.cutout:
        pretrained = args.pretrained+'/cutout_model_best.pth.tar'
    elif args.mixup:
        pretrained = args.pretrained+'/mixup_model_best.pth.tar'
    elif args.cutmix:
        pretrained = args.pretrained+'/cutmix_model_best.pth.tar'
    else:
        pretrained = args.pretrained+'/model_best.pth.tar'

    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(pretrained))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.pretrained))

    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on validation set
    acc, val_loss = validate(val_loader, model, criterion)

    print('Accuracy :', acc)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)

        losses.update(loss.item(), input.size(0))

        acc1.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                i, len(val_loader), loss=losses,acc=acc1))

    return acc1.avg, losses.avg


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


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    _, predicted = torch.max(output.data, 1)  # 得到预测的类别
    total = target.size(0)  # 总的样本数
    correct = (predicted == target).sum().item()  # 预测正确的样本数
    acc = 100 * correct / total
    return acc


if __name__ == '__main__':
    main()