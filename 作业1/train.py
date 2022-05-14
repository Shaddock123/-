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
import utils
import numpy as np
from model import resnet18
from torch.utils.tensorboard import SummaryWriter
from utils import Cutout, rand_bbox, mixup_criterion, mixup_data, save_image_tensor

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='resnet18', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',           # epoch ： 80
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,              # batchszie： 128
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,       # 学习率： 0.1 ，每40个epoch下降10倍
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')

# 用来控制数据增强方式
parser.add_argument('--cutout', type=bool, default=False)
parser.add_argument('--mixup', type=bool, default=True)
parser.add_argument('--cutmix', type=bool, default=False)
parser.add_argument('--vis', type=bool, default=True)

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()  # 解析命令行参数
    if args.cutout:
        print("cutout===========>")
    elif args.mixup:
        print("mixup===========>")
    elif args.cutmix:
        print("cutmix===========>")
    else:
        print("baseline===========>")

    writer = SummaryWriter('./log/')

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    # 如果是cutout模式，就使用下面的数据增强
    if args.cutout:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            Cutout(1,16)                 # 添加cutout方法
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 加载数据
    # 训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    # 测试集
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # 加载模型ResNet18
    print("=> creating model '{}'".format(args.net_type))
    model = resnet18(num_classes=100)
    model = model.cuda()

    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 定义SGD优化器
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True
    # 训练
    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)  # 每40个epoch学习率下降10倍

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # 计算测试集中的精确度以及损失
        acc, val_loss = validate(val_loader, model, criterion, epoch)

        # 将数据添加到tensorboard中，画图
        if args.cutout:
            writer.add_scalar('cutout train loss', train_loss, epoch)
            writer.add_scalar('cutout val loss', val_loss, epoch)
            writer.add_scalar('cutout Acc', acc, epoch)
        elif args.mixup:
            writer.add_scalar('mixup train loss', train_loss, epoch)
            writer.add_scalar('mixup val loss', val_loss, epoch)
            writer.add_scalar('mixup Acc', acc, epoch)
        elif args.cutmix:
            writer.add_scalar('cutmix train loss', train_loss, epoch)
            writer.add_scalar('cutmix val loss', val_loss, epoch)
            writer.add_scalar('cutmix Acc', acc, epoch)
        else:
            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('val loss', val_loss, epoch)
            writer.add_scalar('Acc', acc, epoch)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print('Current best accuracy :', best_acc)

        # 保存模型
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args)

    print('Best accuracy :', best_acc)


# 模型训练
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    img = []
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if args.cutmix:       # cutmix方法
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # if i == 0 and args.vis:
            #     save_image_tensor(input[0], './results/img_cutmix_1.jpg')
            #     save_image_tensor(input[1], './results/img_cutmix_2.jpg')
            #     save_image_tensor(input[2], './results/img_cutmix_3.jpg')
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

        elif args.mixup:     # mixup方法
            inputs, targets_a, targets_b, lam = mixup_data(input, target, use_cuda=True)
            # if i == 0 and args.vis:
            #     save_image_tensor(inputs[0], './results/img_mixup_1.jpg')
            #     save_image_tensor(inputs[1], './results/img_mixup_2.jpg')
            #     save_image_tensor(inputs[2], './results/img_mixup_3.jpg')
            output = model(inputs)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # 计算准确度
        acc = accuracy(output.data, target)

        losses.update(loss.item(), input.size(0))
        acc1.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, loss=losses))

    print('* Epoch: [{0}/{1}]\t acc {acc.avg:.3f}  Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, acc=acc1, loss=losses))

    return losses.avg


# 模型测试验证，在测试集上验证准确率
def validate(val_loader, model, criterion, epoch):
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
        acc = accuracy(output.data, target)  # 计算准确度

        losses.update(loss.item(), input.size(0))

        acc1.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                epoch, args.epochs, i, len(val_loader), loss=losses, acc=acc1))

    print('* Epoch: [{0}/{1}]  acc {acc.avg:.3f} Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, acc=acc1, loss=losses))
    return acc1.avg, losses.avg


# 保存模型参数
def save_checkpoint(state, is_best, args=None):
    if args.cutout:
        filename = 'cutout_checkpoint.pth.tar'
        bestname = 'cutout_model_best.pth.tar'
    elif args.mixup:
        filename = 'mixup_checkpoint.pth.tar'
        bestname = 'mixup_model_best.pth.tar'
    elif args.cutmix:
        filename = 'cutmix_checkpoint.pth.tar'
        bestname = 'cutmix_model_best.pth.tar'
    else:
        filename = 'checkpoint.pth.tar'
        bestname = 'model_best.pth.tar'

    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + bestname)


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


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


# 准确度计算
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    _, predicted = torch.max(output.data, 1)  # 得到预测的类别
    total = target.size(0)  # 总的样本数
    correct = (predicted == target).sum().item()  # 预测正确的样本数
    acc = 100 * correct / total
    return acc


if __name__ == '__main__':
    main()