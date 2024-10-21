import torch
import torch.utils.data as data
import numpy as np
import random

class ParityDataset(data.Dataset):
    def __init__(self, samples, max_bitstring_length):
        self.samples = samples
        self.max_bitstring_length = max_bitstring_length

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        bitstring_length = random.randint(2, self.max_bitstring_length)
        bitstring = torch.randint(0, 2, (bitstring_length,)).to(torch.float32)
        label = torch.tensor([bitstring.sum().item() % 2], dtype=torch.float32)
        return bitstring, label
    
def collate_variable_length(batch):
    bitstrings, labels = zip(*batch)
    lengths = torch.tensor([len(bitstring) for bitstring in bitstrings])
    padded_bitstrings = torch.nn.utils.rnn.pad_sequence(bitstrings, batch_first=True, padding_value = 2)
    labels = torch.stack(labels)
    return {'input_ids': padded_bitstrings.to(torch.long), 'target_ids': labels.to(torch.long), 'lengths': lengths.to(torch.long)}

def parity_dataloaders(batch_size, sequence_length, dataset_size):
    train_size, val_size = int(0.8 * dataset_size), int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset = ParityDataset(train_size, sequence_length)
    val_dataset = ParityDataset(val_size, sequence_length)
    test_dataset = ParityDataset(test_size, sequence_length)
    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: collate_variable_length(x))
    valid_loader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: collate_variable_length(x))
    test_loader = data.DataLoader(test_dataset, batch_size = 32, collate_fn = lambda x: collate_variable_length(x))
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    pass