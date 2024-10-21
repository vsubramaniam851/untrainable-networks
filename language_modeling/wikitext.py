import os
import torch
import torch.utils.data as data
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

class TextDataset(data.Dataset):
    def __init__(self, dataset, num_words):
        self.dataset = dataset
        self.num_words = num_words
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.dataset[idx], 'target_ids': self.dataset[idx]}

def make_dataloaders(batch_size, num_workers = 4, seq_len = 50):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    dataloaders = []
    file_path = 'language_modeling/wikitext-103/wiki.{}.tokens'
    if not os.path.exists(os.path.join('language_modeling', 'cached_wikitext')):
        os.makedirs(os.path.join('language_modeling', 'cached_wikitext'))
    cached_file_path = os.path.join('language_modeling', 'cached_wikitext', 'wikitext_{}.pt')
    for split in ['train', 'valid', 'test']:
        if os.path.exists(cached_file_path.format(split)):
            wikitext_dataset = torch.load(cached_file_path.format(split))
            num_words = wikitext_dataset['num_words']
            dataset = wikitext_dataset['dataset']
        else:
            with open(file_path.format(split), 'r', encoding = 'utf-8') as f:
                dataset = f.readlines()
                num_words = sum([len(line.split(' ')) for line in dataset])
            dataset = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]'))) for line in tqdm(dataset))
            dataset = torch.tensor([index for line in dataset for index in line], dtype = torch.long)
            torch.save({'dataset': dataset, 'num_words': num_words}, cached_file_path.format(split))
        num_sequences = (dataset.size(0) // seq_len) * seq_len
        dataset = dataset.narrow(0, 0, num_sequences).view(-1, seq_len)
        dataset = TextDataset(dataset, num_words)
        dataloader = data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = (split == 'train'))
        dataloaders.append(dataloader)
    return dataloaders, len(tokenizer.vocab)

if __name__ == '__main__':
    pass