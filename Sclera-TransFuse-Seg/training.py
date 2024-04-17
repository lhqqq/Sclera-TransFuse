
from __future__ import print_function
import argparse
import os
import yaml
import shutil
import time
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from cross_efficient_vit import CrossEfficientViT
import timm
import wandb
import cv2
import numpy as np
from load_imglist import ImageList

# https://blog.csdn.net/weixin_43301333/article/details/121155260
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='CNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train_root_path', default='', type=str, metavar='PATH',
                    help='root path of training images (default: none)')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--save_path', default='./checkpoints_recognition_com/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=55, type=int,
                    metavar='N', help='number of classes/subjects')
parser.add_argument('--seed', default=9999999, type=int,
                    metavar='N', help='seed')
parser.add_argument('--config', type=str,
                    help="Which configuration to use. See into 'config' folder.")


def to_rgb(image):
    return image.convert('RGB')

def to_clahe_twice(image):
    img = np.asarray(image)
    img = clahe(img)
    img = clahe(img)
    im = Image.fromarray(img)
    return im

def to_clahe_g_twice(image):
    img = np.asarray(image)
    img = clahe_g(img)
    img = clahe_g(img)
    im = Image.fromarray(img)
    return im

def to_clahe_hsv_twice(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    img = clahe_hsv(img)
    img = clahe_hsv(img)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return im

def clahe(image):
    r, g, b = cv2.split(image)
    # https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)
    result = cv2.merge([r, g, b])
    return result

def clahe_g(image):
    r, g, b = cv2.split(image)
    # https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    result = cv2.merge([g, g, g])
    return result

# https://towardsdatascience.com/increase-your-face-recognition-models-accuracy-by-improving-face-contrast-a3e71bb6b9fb
def clahe_hsv(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():


    set_seed(args.seed)
    with open(args.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    model = CrossEfficientViT(config=config)

    if args.cuda:
        model = model.cuda()

    print(model)

    # large lr for last fc parameters
    params = []
    for name, value in model.named_parameters():
        if 'bias' in name:
            if 'fc3' in name:
                params += [{'params': value, 'lr': 20 * args.lr, 'weight_decay': 0}]
            else:
                params += [{'params': value, 'lr': 2 * args.lr, 'weight_decay': 0}]
        else:
            if 'fc3' in name:
                params += [{'params': value, 'lr': 10 * args.lr}]
            else:
                params += [{'params': value, 'lr': 1 * args.lr}]

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # load image
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.train_root_path, fileList=args.train_list,
                  transform=transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.RandomHorizontalFlip(p=0.5),
                      transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
                      transforms.Lambda(to_clahe_hsv_twice),
                      transforms.ToTensor(),
                  ])),

        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        criterion.cuda()

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        prec1 = train(train_loader, model, criterion, optimizer, epoch)

        save_name = args.save_path + args.model +'_' + str(epoch+1) +'_' + "{:.2f}".format(prec1) + '_checkpoint.pth'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'model': args.model,
            'num_classes': args.num_classes,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }, save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output, _ = model(input)
        loss = criterion(output, target)

        wandb.log({'train/loss': loss.item()})

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        wandb.log({'train/acc': prec1.sum().item()})

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:


            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return top1.avg

def save_checkpoint(state, filename):
    torch.save(state, filename)


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
    scale = 0.457305051927326
    step = 50  # 调
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # import pdb; pdb.set_trace()
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':

    global args
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    wandb.init(
        project=args.model,
        config={
            "learning_rate": args.lr,
            "architecture":  args.arch,
            "epochs": args.epochs,
            'optim': 'SGD',
            'scheduler': '800+0.5'
        }
    )
    main()
    wandb.finish()
