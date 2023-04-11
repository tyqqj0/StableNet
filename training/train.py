import os
import random
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter

from training.reweighting import weight_learner


# 检查每个类别和和哪些特征相关

def compute_correlation(output, cfeatures, batch_size):
    # 计算 output 和 cfeatures 的均值
    output_mean = output.mean(dim=0, keepdim=True)
    cfeatures_mean = cfeatures.mean(dim=0, keepdim=True)

    # 分别计算 output 和 cfeatures 的零均值数据
    output_zero_mean = output - output_mean
    cfeatures_zero_mean = cfeatures - cfeatures_mean

    # 计算 output 和 cfeatures 的协方差
    # 计算 output 和 cfeatures 的协方差
    cov = output_zero_mean.t().matmul(cfeatures_zero_mean) / (batch_size - 1)

    # 计算 output 和 cfeatures 的标准差
    output_std = output_zero_mean.std(dim=0, unbiased=False)
    cfeatures_std = cfeatures_zero_mean.std(dim=0, unbiased=False)

    # 计算相关系数矩阵
    correlation_matrix = cov / (output_std[:, None] * cfeatures_std[None, :])

    return correlation_matrix


def train(train_loader, model, criterion, optimizer, epoch, args, tensor_writer=None):
    ''' TODO write a dict to save previous featrues  check vqvae,
        the size of each feature is 512, os we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time
        TODO init the tensors
    '''
    # model.check()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output, cfeatures = model(images)
        ##################################################
        # 计算相关系数矩阵
        correlation_matrix = compute_correlation(output, cfeatures, args.batch_size)

        # 找到与每个 output 最相关的 cfeatures
        # most_related_cfeatures_indices = correlation_matrix.argmax(dim=1).tolist()

        top_k = 5
        most_related_cfeatures_indices = correlation_matrix.topk(top_k, dim=1).indices.tolist()
        # print(output.shape)
        # 输出结果
        for output_idx, cfeature_idx in enumerate(most_related_cfeatures_indices):
            print(f"Output label{output[output_idx].argmax()} is most related to cfeature {cfeature_idx}")
        ##################################################
        if args.distributed is False and args.gpu is None:
            # 当使用多GPU训练时，模型实际保存位置在model.module中
            # 因此需要从model.module中获取参数
            pre_features, pre_weight1 = model.module.get_prefeatures()
        else:
            pre_features, pre_weight1 = model.get_prefeatures()
        # 输出pre_features, pre_weight1
        # print('pre_features', pre_features[0:10])
        # print('pre_weight1', pre_weight1[0:10])
        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

        if args.distributed is False and args.gpu is None:
            model.module.pre_features.data.copy_(pre_features)
            model.module.pre_weight1.data.copy_(pre_weight1)
        else:
            model.pre_features.data.copy_(pre_features)
            model.pre_weight1.data.copy_(pre_weight1)
        loss = criterion(output, target)
        # print("loss before reweighing\n", loss.shape, '\n', loss)
        loss = loss.view(1, -1).mm(weight1).view(1)  #
        # print("loss after reweighing\n", loss.shape, '\n', loss)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        method_name = args.log_path.split('/')[-2]
        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)

    tensor_writer.add_scalar('loss/train', losses.avg, epoch)
    tensor_writer.add_scalar('ACC@1/train', top1.avg, epoch)
    tensor_writer.add_scalar('ACC@5/train', top5.avg, epoch)
