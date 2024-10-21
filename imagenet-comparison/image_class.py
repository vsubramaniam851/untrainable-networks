import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import warnings
warnings.filterwarnings('ignore')
import wandb

from tqdm import tqdm
from .args import *
from .dataset import ImageNetDataset
from .mlp import ImageNetNarrowMLP, ImageNetShallowMLP
from .resnet_arch import ModifiedResNet50
from rep_sim import rep_similarity_loss
    
def total_loss(train_model, target_model, rep_sim, loss_fn, preds, imgs, labels, rep_sim_alpha, device, student_model = 'ResNet-50', 
               use_noise = False, torchvision_extract = False):
    rep_sim = rep_similarity_loss(train_model, target_model, rep_sim, imgs, device, student_model = student_model, 
                                  use_noise = use_noise, torchvision_extract = torchvision_extract)
    ce_loss = loss_fn(preds, labels)
    return ce_loss + rep_sim_alpha * rep_sim, rep_sim, ce_loss

def get_dataloaders(batch_size = 64, num_workers = 4):
    train_dataset, val_dataset, test_dataset = ImageNetDataset('train'), ImageNetDataset('validation'), ImageNetDataset('test')
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    val_loader = data.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False, pin_memory = True)
    test_loader = data.DataLoader(test_dataset, batch_size = 1)
    return train_loader, val_loader, test_loader

def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(model, val_loader, loss_fn, device):
    model = model.eval()
    val_loss = 0.0
    for batch in tqdm(val_loader, desc = 'Iterating over validation batches...'):
        imgs, labels = batch['image'].to(device), batch['label'].to(device)
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(imgs)
        loss = loss_fn(preds, labels)
        val_loss += loss.item()
    return val_loss/len(val_loader)

def avg_step_size(model, before_state_dict):
    sum_changes = 0
    count = 0
    with torch.no_grad():
        after_state_dict = model.state_dict()
        for key in before_state_dict:
            change = (after_state_dict[key] - before_state_dict[key]).abs().mean().item()
            sum_changes += change
            count += 1
    return sum_changes / count

def load_student_model(student_model, device):
    if student_model == 'WideMLP':
        model = ImageNetShallowMLP()
        model = model.to(device)
    elif student_model == 'NoResNet-50':
        model = ModifiedResNet50()
        model = model.to(device)
    else:
        model = ImageNetNarrowMLP()
        model = model.to(device)

    return model

def train_image_classifier(args, exp_name, rep_sim, num_epochs, student_model = 'ResNet-50', target_model = 'rn50', lr = 1e-3, accumulation = 1, 
                           batch_size = 64, num_workers = 16, pretrained = True, rep_dist = None, rep_sim_alpha = 1.0, use_noise = False):
    wandb.init(
        project = exp_name,
        config = {
            'model': student_model,
            'target_model': target_model,
            'rep-sim': rep_sim,
            'dist-func': rep_dist,
            'lr': lr,
            'batch_size': batch_size,
            'epochs': num_epochs
        }
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, val_loader, _ = get_dataloaders(batch_size, num_workers)
    model = load_student_model(student_model, device)

    target_model_str = target_model
    if rep_sim:
        if target_model == 'rn50':
            target_model = torchvision.models.resnet50(pretrained = pretrained).to(device)
        elif target_model == 'rn18':
            target_model = torchvision.models.resnet18(pretrained = pretrained).to(device)
        elif target_model == 'vitb': 
            target_model = torchvision.models.vit_b_16(pretrained = pretrained).to(device)
        else:
            raise NotImplementedError
        target_model = target_model.eval()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    epoch_train_losses = []
    step_train_losses = []
    step_sizes = []
    val_losses = []
    step_ce_loss = []
    step_rep_sim_loss = []
    accs = []

    total_steps = len(train_loader) * num_epochs

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        adjust_learning_rate(lr, optimizer, epoch)
        avg_val_loss = validate(model, val_loader, loss_fn, device)
        if avg_val_loss < min(val_losses, default = np.nan):
            torch.save(model.state_dict(), f'saved_models/{exp_name}.pt') 
        wandb.log({'val_loss': avg_val_loss})
        print(f'Epoch {epoch}, Validation Loss: {avg_val_loss}')
        val_losses.append(avg_val_loss)

        acc1, acc5 = eval_loop(model, val_loader, device)
        wandb.log({'val_acc1': acc1})
        wandb.log({'val_acc5': acc5})
        print(f'Epoch {epoch}, Validation Accuracy: {acc1}')
        accs.append(acc1)

        model = model.train()
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over training batches...')):
            imgs, labels = batch['image'].to(device), batch['label'].to(device)
            preds = model(imgs)

            if not rep_sim:
                loss = loss_fn(preds, labels)
                ce_loss = None
            else:
                total_step = epoch * len(train_loader) + i
                if args.early_stop and total_step > 300:
                    loss = loss_fn(preds, labels)
                    ce_loss = loss
                    rep_sim = torch.tensor(0)
                else:
                    loss, rep_sim, ce_loss = total_loss(model, target_model, rep_dist, loss_fn, preds, imgs, labels, rep_sim_alpha, device, 
                                                        student_model = student_model, use_noise = use_noise, 
                                                        torchvision_extract = (target_model_str == 'vitb'))
                step_ce_loss.append(ce_loss.item())
                step_rep_sim_loss.append(rep_sim.item())

                if i % 20 == 0:
                    avg_ce_loss = np.mean(step_ce_loss[-20:])
                    avg_rep_sim_loss = np.mean(step_rep_sim_loss[-20:])
                    wandb.log({'ce_loss': avg_ce_loss, 'rep_sim_loss': avg_rep_sim_loss})

            before_update_params = {name: param.clone() for name, param in model.named_parameters()}
            loss.backward()
            if (i + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            if ce_loss == None:
                train_loss += loss.item()
            else:
                train_loss += ce_loss.item()
            step_train_losses.append(loss.item())
            
            step_size = avg_step_size(model, before_update_params)
            step_sizes.append(step_size)
            if i % 20 == 0:
                avg_train_loss = np.mean(step_train_losses[-20:])
                wandb.log({'train_loss': avg_train_loss, 'step_size': step_size})
        
        avg_train_loss = train_loss/len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')
        epoch_train_losses.append(avg_train_loss)

    final_avg_val_loss = validate(model, val_loader, loss_fn, device)
    print(f'Epoch {epoch+1}, Validation Loss: {final_avg_val_loss}')

    assert len(step_train_losses) == len(step_sizes) == total_steps

    if not os.path.exists(f'{args.logging}/{args.exp_name}'):
        os.makedirs(f'{args.logging}/{args.exp_name}')
    
    with open(f'{args.logging}/{exp_name}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    loss_info = {'step_train_losses': step_train_losses, 'step_sizes': step_sizes, 'val_losses': val_losses, 'epoch_train_losses': epoch_train_losses, 
                 'step_ce_loss': step_ce_loss, 'step_rep_sim_loss': step_rep_sim_loss, 'accuracies': accs}
    loss_info = {key: value for key, value in loss_info.items() if value != []}
    with open(f'{args.logging}/{exp_name}/info.json', 'w') as f:
        json.dump(loss_info, f)
    wandb.finish()
    return model, step_train_losses, step_sizes, val_losses, epoch_train_losses, step_ce_loss, step_rep_sim_loss

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = torch.topk(output, maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def eval_loop(model, val_loader, device):
    model = model.eval()
    top1_acc = 0
    top5_acc = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc = 'Iterating over test batches...'):
            img, label = batch['image'].to(device), batch['label'].to(device)
            outputs = model(img)
            acc1, acc5 = accuracy(outputs, label, topk=(1, 5))
            top1_acc += acc1.item() * img.size(0)
            top5_acc += acc5.item() * img.size(0)
            total_samples += img.size(0)
    top1_acc /= total_samples
    top5_acc /= total_samples
    return top1_acc, top5_acc

def eval_image_classifier(exp_name, student_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if student_model == 'WideMLP':
        model = ImageNetShallowMLP()
    elif student_model == 'DeepMLP':
        model = ImageNetNarrowMLP()
    elif student_model == 'NoResNet-50':
        model = ModifiedResNet50()
    else:
        raise NotImplementedError()
    model.load_state_dict(torch.load(f'saved_models/{exp_name}.pt'))
    model = model.to(device)
    _, val_loader, _ = get_dataloaders(1, 4)

    top1_acc, top5_acc = eval_loop(model, val_loader, device)
    print(f'Top-1 Accuracy: {top1_acc:.2f}%')
    print(f'Top-5 Accuracy: {top5_acc:.2f}%')
            
if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.eval:
        model, step_train_losses, step_sizes, val_losses, epoch_train_losses, step_ce_loss, step_rep_sim_loss = train_image_classifier(args, args.exp_name, args.rep_sim, args.num_epochs, 
                                                                                                                                       student_model = args.student_model, target_model = args.target_model, lr = args.lr, 
                                                                                                                                       accumulation = args.accumulation, batch_size = args.batch_size, num_workers = args.num_workers, 
                                                                                                                                       pretrained = args.pretrained, rep_dist = args.rep_dist, rep_sim_alpha = args.rep_sim_alpha, 
                                                                                                                                       use_noise = args.use_noise)
    else:
        eval_image_classifier(args.exp_name, args.student_model)