import os
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from yaml import load

from modules import *
from utils import generate_model, print_msg
from utils import print_msg, inverse_normalize

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()     #k*batchsize
        correct = pred.eq(target.view(1, -1).expand_as(pred))   #5*40

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(model, train_config, data_config, train_dataset,val_dataset, save_path, device,checkpoint=None, logger=None):

    optimizer = initialize_optimizer(train_config, model)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(train_config, optimizer)
    start_epoch = 1

    if checkpoint:#模型已经在main.py里加载过了
        if os.path.exists(checkpoint):

            ck = torch.load(checkpoint)
            if ck.get('optimizer',None):
                optimizer.load_state_dict(ck['optimizer'].state_dict())
            if ck.get('lr_scheduler',None):
                lr_scheduler.load_state_dict(ck['lr_scheduler'].state_dict())
            if ck.get('warmup_scheduler') and warmup_scheduler:
                warmup_scheduler.load_state_dict(ck['warmup_scheduler'].state_dict())
            if ck.get('epoch'):
                start_epoch = ck['epoch']
        else:
            print('Don`t find checkpoint model!')
        

    # loss_function = ContrastiveLoss().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    train_loader,val_loader = initialize_dataloader(train_config, train_dataset,val_dataset)   #已经经过数据增强的数据
    patchsize = data_config['patch_size']
    # start training
    
    model.train()
    min_indicator = 9999999    #用来记录最小loss
    avg_loss = 0
    for epoch in range(start_epoch, train_config['epochs'] + 1):
        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        avg_acc1,avg_acc5 = 0,0
        top1,top5 = 0,0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            #X_1,X_2是lesion patch，H是healthy patch
            X_1, X_2,H = train_data    #(4,3,128,128)   batchsize,c,h,w
            H = H.view(-1,3,patchsize,patchsize)

            X_1,X_2,H = X_1.to(device),X_2.to(device),H.to(device)

            # forward  
            output,target = model(X_1,X_2,H)   #batchsize*(K+1) , batchsize
            loss = loss_function(output,target)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1 += acc1.item()
            avg_acc1 = top1 / (step + 1)
            top5 += acc5.item()
            avg_acc5 = top5 / (step + 1)
            progress.set_description(
                'epoch: {}, loss: {:.6f}, acc1:{:.6f}, acc5:{:.6f}'
                .format(epoch, avg_loss, avg_acc1, avg_acc5)
            )

        if epoch % 10 == 0:
            val_loss = eval(model, val_loader,loss_function, device)
            logger.add_scalar('validation loss', val_loss, epoch)
            print('validation loss: {:.6f}'.format(val_loss))

            # save model
            indicator = val_loss
            if indicator < min_indicator:
                torch.save({'epoch':epoch+1,
                            'optimizer':optimizer.state_dict(),
                            'model':model.state_dict(),
                            'lr_scheduler':lr_scheduler,
                            'warmup_scheduler':warmup_scheduler}, 
                            os.path.join(save_path, 'best_validation_model.pt'))
                min_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(save_path))

        if epoch % train_config['save_interval'] == 0:
            torch.save({'epoch':epoch+1,
                            'optimizer':optimizer.state_dict(),
                            'model':model.state_dict(),
                            'lr_scheduler':lr_scheduler,
                            'warmup_scheduler':warmup_scheduler}, 
                            os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if train_config['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        curr_lr = optimizer.param_groups[0]['lr']
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    torch.save({'epoch':epoch+1,
                            'optimizer':optimizer,
                            'model':model.state_dict(),
                            'lr_scheduler':lr_scheduler,
                            'warmup_scheduler':warmup_scheduler}, 
                            os.path.join(save_path, 'final_model.pt'))
    if logger:
        logger.close()

def evaluate(model_path, train_config, test_dataset, device):
# def evaluate(model_path, train_config, test_dataset, num_classes, estimator, device):
    trained_model = torch.load(model_path).to(device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        shuffle=False
    )

    print('Running on Test set...')
    
    # eval(trained_model, test_loader, train_config['criterion'], estimator, device)
    eval(trained_model, test_loader, train_config['criterion'], device)


def eval(model, dataloader, loss_function, device):
    model.eval()
    torch.set_grad_enabled(False)

    # estimator.reset()
    val_loss = 0
    avg_loss = 0
    for step, test_data in enumerate(dataloader):
        X_1, X_2 = test_data
        X_1,X_2 = X_1.to(device),X_2.to(device)
        
        output,target = model(X_1,X_2,None,is_train=False)
        loss = loss_function(output,target)

        val_loss += loss.item()
        avg_loss = val_loss / (step + 1)

    model.train()
    torch.set_grad_enabled(True)
    return avg_loss



# define data loader
def initialize_dataloader(train_config, train_dataset, val_dataset):
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    pin_memory = train_config['pin_memory']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )


    return train_loader,val_loader


# define optmizer
def initialize_optimizer(train_config, model):
    optimizer_strategy = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    nesterov = train_config['nesterov']
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(train_config, optimizer):
    learning_rate = train_config['learning_rate']
    warmup_epochs = train_config['warmup_epochs']
    scheduler_strategy = train_config['lr_scheduler']
    scheduler_config = train_config['scheduler_config']

    lr_scheduler = None
    if scheduler_strategy in scheduler_config.keys():
        scheduler_config = scheduler_config[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
