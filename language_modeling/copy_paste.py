import numpy as np
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn

class CopyPasteDataset(data.Dataset):
    def __init__(self, num_samples, seq_min_len, seq_max_len, vocab_size):
        self.num_samples = num_samples
        self.seq_min_len = seq_min_len
        self.seq_max_len = seq_max_len
        self.vocab_size = vocab_size
        self.copy_token = vocab_size + 1
        self.eos_token = vocab_size + 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_len = np.random.randint(self.seq_min_len, self.seq_max_len + 1)
        sequence = np.random.randint(0, self.vocab_size, seq_len)

        input_sequence = np.concatenate([sequence, [self.copy_token], np.zeros(seq_len), [self.eos_token]])
        target_sequence = np.concatenate([np.zeros(seq_len + 1), sequence, [self.eos_token]])

        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)

        return input_tensor, target_tensor 

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return {'input_ids': inputs_padded, 'target_ids': targets_padded}

def get_dataloaders(batch_size, num_samples, seq_min_len, seq_max_len, vocab_size):
    train_dataset, val_dataset = CopyPasteDataset(int(0.8 * num_samples), seq_min_len, seq_max_len, vocab_size), CopyPasteDataset(num_samples - int(0.8 * num_samples), seq_min_len, seq_max_len, vocab_size)
    test_dataset = CopyPasteDataset(len(val_dataset), seq_min_len, seq_max_len, vocab_size)

    train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    test_dataloader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

    return train_dataloader, val_dataloader, test_dataloader
