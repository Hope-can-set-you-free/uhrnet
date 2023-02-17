import os
from config import *
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, Gpu_list))  # 指定使用的GPU
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from d2l import torch as d2l
from nets.uhrnet import UHRnet
from nets.uhrnet_training import weights_init
from utils.dataloader import SegmentationDataset, seg_dataset_collate
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.utils import seed_torch
from utils.utils import init_linear_weights
from nets.uhrnet_training import CE_Loss, Dice_loss, Focal_Loss, weights_init, get_lr_scheduler, set_optimizer_lr
from tqdm import tqdm
from utils.utils import show_config
from utils.utils import get_lr
from utils import plt_pictures as pl
from time import time
import pandas as pd


def train_model_flc(model, VOCdevkit_path, train_txt_path, val_txt_path, Init_lr, momentum, weight_decay,
                    batch_size, input_shape, num_classes, num_workers, total_Epoch
                    ):

    with open(os.path.join(VOCdevkit_path, train_txt_path),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, val_txt_path),"r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    epoch_step = num_train // (batch_size * len(Gpu_list))
    epoch_step_val = num_val // (batch_size * len(Gpu_list))

    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
    val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    gen = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate)

    animator_loss = pl.Animator(xlabel='epoch',
                                ylabel='loss',
                                legend=['train_loss', 'val_loss'],
                                xlim=[1, total_Epoch], figsize=(14, 10))


    warmup_total_iters = epoch_step * warmup_epoch
    total_iters = epoch_step * total_Epoch

    if local_rank == 0:
        show_config(
            Init_lr = Init_lr,
            momentum = momentum,
            weight_decay = weight_decay,
            batch_size = batch_size,
            input_shape = input_shape,
            num_classes = num_classes,
            num_workers = num_workers,
            total_Epoch = total_Epoch,
            num_train = num_train,
            num_val = num_val,
            epoch_step = epoch_step,
            epoch_step_val = epoch_step_val
        )
        total_time_begin = time()

    train_loss_list = []
    test_loss_list = []
    for epoch in range(total_Epoch):
        if local_rank == 0:
            each_time_begin = time()
        if local_rank == 0:
            print('Start Train')
        total_loss = 0
        model.train()
        dist.barrier()
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{total_Epoch}', postfix=dict, mininterval=0.3) as pbar:
            iters_now = 0
            for iteration, batch in enumerate(gen):
                iters_now += 1
                iters = epoch * epoch_step + iters_now
                if iteration >= epoch_step:
                    break
                imgs, pngs, labels = batch
                with torch.no_grad():
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                    pngs = torch.from_numpy(pngs).long()
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)
                    weights = torch.from_numpy(cls_weights)
                    imgs = imgs.cuda(local_rank)
                    pngs = pngs.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

                optimizer.zero_grad()
                outputs = model(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                loss.backward()
                optimizer.step()

                if local_rank == 0:
                    # pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                    #                     'f_score': total_f_score / (iteration + 1),
                    #                     'lr': get_lr(optimizer)})
                    pbar.update(1)

                # 设置学习率
                if lr_strategy == 'warmup':
                    lr_scheduler_func = get_lr_scheduler(Init_lr, total_iters, warmup_total_iters, warmup_lr_start)
                    set_optimizer_lr(optimizer, lr_scheduler_func, iters)

                if lr_strategy == 'step':
                    Init_lr *= gamma
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = Init_lr

        if local_rank == 0:
            print('Finish Train')
            print('Start Validation')

        dist.barrier()
        model.eval()

        loss_eval = 0
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{total_Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                imgs, pngs, labels = batch
                with torch.no_grad():
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                    pngs = torch.from_numpy(pngs).long()
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)
                    weights = torch.from_numpy(cls_weights)

                    imgs = imgs.cuda(local_rank)
                    pngs = pngs.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

                    outputs = model(imgs)
                    if focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        loss = loss + main_dice

                    loss_eval += loss.item()
                    loss_eval /= num_val

                # if local_rank == 0:
                #     print('loss : ', loss)
                #     print('loss.item() : ', loss.item())
                #     print('loss_eval : ', loss_eval)


                if local_rank == 0:
                    pbar.update(1)

        test_loss_list.append(loss_eval)




        loss_train = 0
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{total_Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                imgs, pngs, labels = batch
                with torch.no_grad():
                    imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                    pngs = torch.from_numpy(pngs).long()
                    labels = torch.from_numpy(labels).type(torch.FloatTensor)
                    weights = torch.from_numpy(cls_weights)

                    imgs = imgs.cuda(local_rank)
                    pngs = pngs.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

                    outputs = model(imgs)
                    if focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        loss = loss + main_dice

                    loss_train += loss.item()
                    loss_train /= num_train

                # if local_rank == 0:
                #     print('loss : ', loss)
                #     print('loss.item() : ', loss.item())
                #     print('loss_train : ', loss_train)


                if local_rank == 0:
                    pbar.update(1)

        train_loss_list.append(loss_train)

        animator_loss.add(epoch + 1, (loss_train, loss_eval))

        if local_rank == 0:
            print('Finish Validation')

        if (epoch + 1) % save_period == 0 or epoch + 1 == total_Epoch:
            if local_rank == 0:
                print('save succeed!!!!!!!!')
            torch.save(model.state_dict(), 'logs/ep%03d.pth' % (
            epoch + 1))


        if local_rank == 0:
            each_time_end = time()
            print('Epoch:' + str(epoch + 1) + '/' + str(total_Epoch)  + '_runtime : ', each_time_end - each_time_begin)


    if local_rank == 0:
        animator_loss.save('logs/ep%03d.png' % (
        epoch + 1))
        total_time_end = time()
        df = dict()
        df['Init_lr'] = Init_lr
        df['momentum'] = momentum
        df['weight_decay'] = weight_decay
        df['batch_size'] = batch_size
        df['input_shape'] = input_shape
        df['total_Epoch'] = total_Epoch
        df['num_train'] = num_train
        df['local_rank'] = local_rank
        df['num_val'] = num_val
        df['batch_size'] = batch_size
        df['seed'] = seed
        df['num_workers'] = num_workers
        df['epoch_step_val'] = epoch_step_val
        df['total_time'] = total_time_end - total_time_begin
        df = pd.DataFrame([df]).T
        # df.to_excel(os.path.join(path1, 'docum.xlsx'))
        df.to_excel('logs/ep%03d_config.xlsx' % (
                epoch + 1))

        df1 = pd.DataFrame()
        df1['loss_train'] = train_loss_list
        df1['loss_test'] = test_loss_list

        df1.to_excel('logs/ep%03d_loss.xlsx' % (
                epoch + 1))

        # df1.to_excel(os.path.join(path1, 'acc_loss.xlsx'))
        # shutil.copy('config.py', os.path.join(path1, 'config.py'))



if __name__ == '__main__':
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group('nccl', world_size=len(Gpu_list), rank=local_rank)
    torch.backends.cudnn.benchmark = benchmark
    seed_torch(seed)
    model = UHRnet(num_classes=num_classes, backbone=backbone)
    if sync_bn and len(Gpu_list) > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.apply(init_linear_weights)
    model = nn.parallel.DistributedDataParallel(model.cuda(local_rank), device_ids=[local_rank], find_unused_parameters=True)

    train_model_flc(model, VOCdevkit_path, train_txt_path, val_txt_path, Init_lr, momentum, weight_decay,
                    batch_size, input_shape, num_classes, num_workers, total_Epoch)



