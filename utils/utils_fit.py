import torch
from nets.uhrnet_training import CE_Loss, Dice_loss, Focal_Loss, weights_init, get_lr_scheduler, set_optimizer_lr
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
import torch.distributed as dist


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, save_period, lr_strategy, gamma, warmup_total_iters, warmup_lr_start, total_iters, start_lr, local_rank=0):
    gamma           = 0.92
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        print('epoch_step : ', epoch_step)
        print('epoch : ', epoch)
        print('Epoch : ', Epoch)
        # pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()

    iters_now = 0
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            iters_now += 1
            iters = epoch * epoch_step + iters_now
            if iteration >= epoch_step: 
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs    = torch.from_numpy(pngs).long()
                labels  = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)

                if cuda:
                    imgs = imgs.cuda(local_rank)
                    pngs = pngs.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

            optimizer.zero_grad()

            outputs = model_train(imgs)

            # from torchviz import make_dot
            # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # # modelviz = SRCNN().to(device)
            # # input = torch.rand(8, 1, 8, 8).to(device)
            # # out = modelviz(input)
            # # print(out.shape)
            #
            # # 1. ?????? torchviz ?????????
            # g = make_dot(outputs)
            # g.view()

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   ??????f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_f_score   += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'f_score': total_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

            #???????????????
            if lr_strategy == 'warmup':
                lr_scheduler_func = get_lr_scheduler(start_lr, total_iters, warmup_total_iters, warmup_lr_start)
                set_optimizer_lr(optimizer, lr_scheduler_func, iters)

            if lr_strategy == 'step':
                lr *= gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    dist.barrier()
    model_train.eval()
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs    = torch.from_numpy(pngs).long()
                labels  = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda(local_rank)
                    pngs = pngs.cuda(local_rank)
                    labels = labels.cuda(local_rank)
                    weights = weights.cuda(local_rank)

                outputs     = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss  = loss + main_dice
                #-------------------------------#
                #   ??????f_score
                #-------------------------------#
                _f_score    = f_score(outputs, labels)

                val_loss    += loss.item()
                val_f_score += _f_score.item()
            if local_rank == 0:
                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
    if local_rank == 0:
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

    if local_rank == 0:
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
