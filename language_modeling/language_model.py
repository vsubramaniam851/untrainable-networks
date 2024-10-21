import os
import numpy as np
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import wandb

from tqdm import tqdm
from .args import *
from .rnn_models import RNNLM, ParityRNN
from .transformer import Transformer, TransEnc, TransformerLM
from .wikitext import make_dataloaders
from .copy_paste import get_dataloaders
from .parity import parity_dataloaders
from rep_sim import rep_similarity_loss

def total_loss(train_model, target_model, rep_sim, loss_fn, preds, inputs, labels, rep_sim_alpha, device, student_model = 'LSTM', 
               lengths = None, use_noise = False):
    sim_loss = rep_similarity_loss(train_model, target_model, rep_sim, inputs, device, student_model = student_model, 
                                   lengths = lengths, use_noise = use_noise)
    ce_loss = loss_fn(preds, labels)
    return ce_loss + rep_sim_alpha * sim_loss, sim_loss, ce_loss

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

def cp_acc(test_loader, model, device):
    avg_acc = []
    for batch in tqdm(test_loader, desc = 'Evaluating copy-paste accuracy...'):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        with torch.no_grad():
            cp_logits, hidden = model(input_ids, hidden = None)

        preds = torch.argmax(cp_logits, dim = -1)
        pred_flat = preds.view(-1)
        target_flat = target_ids.view(-1)
        mask = (target_flat != 0)
        
        correct = torch.sum(pred_flat[mask] == target_flat[mask]).item()        
        total = mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_acc.append(accuracy)
    return np.mean(avg_acc), np.std(avg_acc, ddof=1) / np.sqrt(len(avg_acc))

def parity_acc(test_loader, model, device):
    correct = 0
    total = 0
    model = model.eval()
    for batch in tqdm(test_loader, desc = 'Evaluating parity accuracy...'):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device).squeeze()
        lengths = batch['lengths'].to(device)

        with torch.no_grad():
            parity_logits, _ = model(input_ids, lengths = lengths, hidden = None)
        parity_probs = torch.sigmoid(parity_logits)
        parity_preds = torch.argmax(parity_probs, dim = -1).squeeze()
        num_correct = (parity_preds == target_ids).sum()
        correct += num_correct.item()
        total += parity_preds.shape[0]
    return correct / total

def shift_logits_labels(task, input_ids, lm_logits, batch, lengths, device):
    if task == 'next-word':
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = input_ids[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
    elif task == 'copy-paste':
        shift_logits = lm_logits.clone()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = batch['target_ids'].clone().to(device)
        shift_labels = shift_labels.view(-1)
    else:
        targets = batch['target_ids'].to(device).squeeze()
        lengths = lengths.to(device)
        shift_labels = torch.nn.functional.one_hot(targets, num_classes=2).float().squeeze()
        shift_logits = lm_logits.clone()
    return shift_logits, shift_labels, lengths

def load_student_model(student_model, vocab_size, task, embedding_dim, hidden_dim, num_layers, fc_dim, nhead, context_length, 
                       device):
    if student_model == 'LSTM':
        model = RNNLM(student_model, vocab_size = vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
                      num_layers = num_layers, fc_dim = fc_dim, device = device)
    elif student_model == 'RNN':
        model = RNNLM(student_model, vocab_size = vocab_size, embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
                      num_layers = num_layers, fc_dim = fc_dim, device = device)
    elif student_model == 'Transformer':
        if task == 'copy-paste':
            model = Transformer(vocab_size, d_model = hidden_dim, nhead = nhead, num_layers = num_layers, dim_feedforward = fc_dim)
        else:
            model = TransformerLM(vocab_size, d_model = hidden_dim, nhead = nhead, num_layers = num_layers, dim_feedforward = fc_dim, seq_len = context_length)
    elif student_model == 'TransEnc':
        model = TransEnc(vocab_size = vocab_size, d_model = hidden_dim, nhead = nhead, num_encoder_layers = num_layers, device = device)
    elif student_model == 'ParityRNN':
        model = ParityRNN(vocab_size = vocab_size, embedding_dim = vocab_size, hidden_dim = hidden_dim, num_layers = num_layers, device = device)
    else:
        raise NotImplementedError()
    model = model.to(device)
    return model

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_lm(args, exp_name, task, repr_sim, student_model, target_model = 'lstm', target_model_name = None, pretrained = True, context_length = 256, num_epochs = 10, 
             batch_size = 64, lr = 1e-3, accumulation = 1, embedding_dim = 256, hidden_dim = 512, num_layers = 4, fc_dim = 512, nhead = 4,
             rep_dist = None, rep_sim_alpha = 1.0, use_noise = False):
    wandb.init(
        project = exp_name,
        config = {
            'model': student_model,
            'repr-sim': repr_sim,
            'lr': lr,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'fc_dim': fc_dim,
            'rep_dist': rep_dist
        }
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if task == 'next-word':
        [train_loader, valid_loader, test_loader], vocab_size = make_dataloaders(batch_size, seq_len = context_length)
        loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
    elif task == 'copy-paste':
        train_loader, valid_loader, test_loader = get_dataloaders(batch_size, num_samples = args.num_samples, seq_min_len = 20, seq_max_len = 40, vocab_size = 10)
        vocab_size = 13
        loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    elif task == 'parity':
        train_loader, valid_loader, test_loader = parity_dataloaders(batch_size, sequence_length = 50, dataset_size = args.num_samples)
        vocab_size = 3
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError()
    
    model = load_student_model(student_model, vocab_size, task, embedding_dim, hidden_dim, num_layers, fc_dim, nhead, context_length,
                               device)

    if repr_sim:
        if target_model == 'transformer':
            if task == 'copy-paste':
                target_model = Transformer(vocab_size, d_model = 512, nhead = 16, num_layers = 4, dim_feedforward = fc_dim)
                if pretrained: 
                    target_model.load_state_dict(torch.load(f'saved_models/{target_model_name}'))
            else:
                target_model = TransformerLM(vocab_size = vocab_size, d_model = 512, nhead = 16, num_layers = 4, seq_len = 256, dim_feedforward = 2048)
                if pretrained:
                    target_model.load_state_dict(torch.load(f'saved_models/{target_model_name}'))
            target_model = target_model.to(device)
        elif target_model == 'parity_rnn':
            target_model = ParityRNN(vocab_size, embedding_dim = vocab_size, hidden_dim = hidden_dim, num_layers = num_layers, device = device)
            if pretrained:
                target_model.load_state_dict(torch.load(f'saved_models/{target_model_name}'))
            target_model = target_model.to(device)
        else:
            raise NotImplementedError

    optimizer = optim.AdamW(model.parameters(), lr = lr)
    val_losses = []
    epoch_train_losses = []
    step_train_losses = []
    step_sizes = []
    step_ce_loss = []
    step_rep_sim_loss = []
    accs = []

    for epoch in range(num_epochs):
        model.eval()
        valid_loss = 0.0
        hidden = None
        for batch in tqdm(valid_loader, desc = 'Iterating over validation set...'):
            input_ids = batch['input_ids'].to(device)
            lengths = batch.get('lengths', None)
            with torch.no_grad():
                lm_logits, hidden = model(input_ids, hidden = None, lengths = lengths)
            shift_logits, shift_labels, lengths = shift_logits_labels(task, input_ids, lm_logits, batch, lengths, device)
            loss = loss_fn(shift_logits, shift_labels)
            valid_loss += loss.item()
        avg_val_loss = valid_loss / len(valid_loader)
        wandb.log({'val_loss': avg_val_loss})
        val_losses.append(avg_val_loss)
        if avg_val_loss <= min(val_losses):
            torch.save(model.state_dict(), f'saved_models/{exp_name}.pt')
        print(f'Epoch {epoch + 1}, Validation Loss {avg_val_loss}')

        if task == 'copy-paste':
            acc = cp_acc(test_loader, model, device)
        elif task == 'parity':
            print('Validation Task Accuracy', parity_acc(valid_loader, model, device))
            acc = parity_acc(test_loader, model, device)
        else:
            acc = eval_ppl_runner(model, test_loader, device)
        print(f'Task Accuracy {acc}')
        accs.append(acc)

        model.train()
        train_loss = 0.0
        hidden = None
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over train loader')):
            input_ids = batch['input_ids'].to(device)
            lengths = batch.get('lengths', None)
            lm_logits, hidden = model(input_ids, hidden = None, lengths = lengths)
            shift_logits, shift_labels, lengths = shift_logits_labels(task, input_ids, lm_logits, batch, lengths, device)
            if not repr_sim:
                loss = loss_fn(shift_logits, shift_labels)
                ce_loss = None
            else:
                loss, sim_loss, ce_loss = total_loss(model, target_model, rep_dist, loss_fn, shift_logits, input_ids, shift_labels, rep_sim_alpha, 
                                                    device, student_model = student_model, lengths = lengths, use_noise = use_noise)
                step_ce_loss.append(ce_loss.item())
                step_rep_sim_loss.append(sim_loss.item())
                if i % 20 == 0:
                    avg_ce_loss = np.mean(step_ce_loss[-20:])
                    avg_rep_sim_loss = np.mean(step_rep_sim_loss[-20:])
                    wandb.log({'ce_loss': avg_ce_loss, 'rep_sim_loss': avg_rep_sim_loss})
            before_update_params = {name: param.clone() for name, param in model.named_parameters()}
            loss.backward()   
            grad_norm = get_grad_norm(model)
            wandb.log({'grad_norm': grad_norm})
            if task == 'next-word':
                nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.25)
            if (i + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            if ce_loss == None:
                train_loss += loss.item()
            else:
                train_loss += ce_loss.item()
            step_train_losses.append(loss.item())
            hidden = tuple(h.detach() for h in hidden)

            step_size = avg_step_size(model, before_update_params)
            step_sizes.append(step_size)

            if i % 20 == 0:
                avg_train_loss = np.mean(step_train_losses[-20:])
                wandb.log({'train_loss': avg_train_loss if task == 'next-word' else avg_train_loss, 'step_size': step_size})
        avg_train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}, Training Loss {avg_train_loss}')

    if not os.path.exists(f'{args.logging}/{args.exp_name}'):
        os.makedirs(f'{args.logging}/{args.exp_name}')
    with open(f'{args.logging}/{exp_name}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    loss_info = {'step_train_losses': step_train_losses, 'step_sizes': step_sizes, 'val_losses': val_losses, 'epoch_train_losses': epoch_train_losses, 'step_ce_loss': step_ce_loss, 'step_rep_sim_loss': step_rep_sim_loss, 'accuracies': accs}
    loss_info = {key: value for key, value in loss_info.items() if value != []}
    with open(f'{args.logging}/{exp_name}/info.json', 'w') as f:
        json.dump(loss_info, f)

    wandb.finish()
    return model, epoch_train_losses, step_train_losses, val_losses, step_ce_loss, step_rep_sim_loss, accs

def load_eval_model(args, task, exp_name, device, student_model, vocab_size):
    with open(os.path.join(args.logging, exp_name, 'args.json'), 'r') as f:
        model_args = json.load(f)
    if student_model == 'RNN':
        model = RNNLM(student_model, vocab_size = vocab_size, embedding_dim = model_args['embedding_dim'], hidden_dim = model_args['hidden_dim'], num_layers = model_args['num_layers'], fc_dim = model_args['fc_dim'], device = device)
    elif student_model == 'Transformer':
        if task == 'copy-paste':
            model = Transformer(vocab_size = vocab_size, d_model = model_args['hidden_dim'], nhead = model_args['nhead'], num_layers = model_args['num_layers'], dim_feedforward = model_args['fc_dim'])
        else:
            model = TransformerLM(vocab_size, d_model = args['hidden_dim'], nhead = model_args['nhead'], num_layers = model_args['num_layers'], dim_feedforward = model_args['fc_dim'], seq_len = model_args['context_length'])
    else:
        raise NotImplementedError()

    state_dict = torch.load(f'saved_models/{exp_name}.pt')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = model.eval()
    return model, model_args['context_length']

def eval_ppl_runner(model, test_loader, device):
    criterion = nn.CrossEntropyLoss(ignore_index = -1)
    model = model.eval()
    total_loss = 0.0
    for batch in tqdm(test_loader, desc = 'Evaluating perplexity...'):
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            lm_logits, hidden = model(input_ids, hidden = None)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.item()
        total_loss += loss
    average_nll = total_loss / len(test_loader)
    word_nll = average_nll * test_loader.dataset.dataset.numel() / test_loader.dataset.num_words
    return math.exp(word_nll)

def eval_ppl(args, exp_name, student_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    [_, val_loader, test_loader], vocab_size = make_dataloaders(32, max_length = 75)
    model, context_length = load_eval_model(args, 'next-word', exp_name, device, student_model, vocab_size)
    return eval_ppl_runner(model, test_loader, device)

def copy_paste_acc(args, exp_name, student_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    _, _, test_loader = get_dataloaders(32, num_samples = 100000, seq_min_len = 20, seq_max_len = 40, vocab_size = 1000)
    model, _ = load_eval_model(args, 'copy-paste', exp_name, device, student_model, 13)
    return cp_acc(test_loader, model, device)

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.eval:
        model, epoch_train_losses, step_train_losses, val_losses, step_ce_loss, step_rep_sim_loss, accs = train_lm(args, args.exp_name, args.task, args.rep_sim, 
                                                                                                                args.student_model, args.target_model, args.target_model_name, args.pretrained, args.context_length, args.num_epochs, 
                                                                                                                args.batch_size, args.lr, args.accumulation, args.embedding_dim, args.hidden_dim, args.num_layers, 
                                                                                                                args.fc_dim, args.nheads, args.rep_dist, args.rep_sim_alpha,
                                                                                                                use_noise = args.use_noise, multi_gpu = args.multi_gpu)
    else:
        if args.task == 'copy-paste':
            print(copy_paste_acc(args, args.exp_name, args.student_model))
        else:
            print(eval_ppl(args, args.exp_name, args.student_model))