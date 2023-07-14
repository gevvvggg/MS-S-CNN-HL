from __future__ import print_function
from losses import SupConLoss
import sys
import argparse
import time
import math
import os
import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
import tensorboard_logger as tb_logger
from torch import nn
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='1000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default=r'D:\PycharmProjects\pythonProject\SupContrast-master\SupContrast-master\save\SupCon\cifar10_models\f2v3.1_SimCLR_cifar10_resnet18_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine\ckpt_epoch_10.pth',
                        help='path to pre-trained model')
    parser.add_argument('--flag', type=int, default='0',
                        help='flag')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'linear_{}_{}_lr_{}_decay_{}_bsz_{}_zhegai'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 4
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    # classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    #     else:
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             k = k.replace("module.", "")
    #             new_state_dict[k] = v
    #         state_dict = new_state_dict
    #     model = model.cuda()
    #     # classifier = classifier.cuda()
    #     criterion = criterion.cuda()
    #     cudnn.benchmark = True
    #
    #     model.load_state_dict(state_dict)
    #     model.fc=nn.Linear(model.head[0].in_features,opt.n_cls)
    #     nn.init.xavier_uniform_(model.fc.weight)
    #     model.head=nn.Sequential()
    # model.load_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.head = nn.Sequential()
    model = model.cuda()
    # classifier = classifier.cuda()
    criterion = criterion.cuda()
    return model,criterion


def train(train_loader, model, criterion,criterion1, optimizer, epoch, opt,classifier):
    """one epoch training"""
    model.train()
    # classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images,images1, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        images1 = images1.cuda(non_blocking=True)
        bsz = labels.shape[0]
        #新加的contrastiveloss
        z = torch.cat([images, images1], dim=0)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        #新加的contrastiveloss
        features1 = model(z)
        f1, f2 = torch.split(features1, [bsz, bsz], dim=0)
        features1 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # compute loss
        # with torch.no_grad():
        #     features = model.encoder(images)
        # output = classifier(features.detach())

        output = model(images)
        output = classifier(output)
        loss = criterion(output, labels)
        #新家的contrastiveloss
        newloss=criterion1(features1)
        loss1=0*newloss+1*loss
        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt,classifier):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            # images = images.float()
            # labels = labels
            bsz = labels.shape[0]

            # forward
            output = model(images)
            output = classifier(output)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # build data loader
    train_loader, val_loader = set_loader(opt)
    #新家的loss
    criterion1 = SupConLoss(temperature=opt.temp)
    # build model and criterion
    model, criterion = set_model(opt)
    classifier = nn.Linear(512, 15)
    classifier = classifier.cuda()

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, criterion,criterion1,
                          optimizer, epoch, opt,classifier)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch%10==0:
            # eval for one epoch
            loss, val_acc = validate(val_loader, model, criterion, opt, classifier)
            logger.log_value('val_loss', loss, epoch)
            logger.log_value('val_acc', val_acc, epoch)
            # if best_acc == 0:
            #     torch.save(model, 'D:\sardata\moxing\pre0.pth')
            #     torch.save(classifier, 'D:\sardata\moxing\lin0.pth')
            if val_acc > best_acc:
                best_acc = val_acc
                print(best_acc)
                torch.save(model, r'D:\sardata\moxing\f2v7.pth')
                torch.save(classifier, r'D:\sardata\moxing\f2vn7.pth')
                # if best_acc >= 93.5:
                #     torch.save(model, r'D:\sardata\moxing\n1.pth')
                #     torch.save(classifier, r'D:\sardata\moxing\n2.pth')
                #     break
                # elif best_acc >= 96 and best_acc <= 96.5:
                #     torch.save(model, 'D:\sardata\moxing\Pnpre96.pth')
                #     torch.save(classifier, 'D:\sardata\moxing\Pnlin96.pth')
                #     break
                # else:
                #     pass
            # if epoch % opt.save_freq == 0:
            #     save_file = os.path.join(
            #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #     save_model(model, optimizer, opt, epoch, save_file)

            # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)
    # print('best accuracy: {:.2f}'.format(best_acc))
    print(best_acc)

if __name__ == '__main__':
    main()
