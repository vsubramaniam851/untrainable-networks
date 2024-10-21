import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

class RNNLM(nn.Module):
    def __init__(self, rnn_model, vocab_size, embedding_dim, hidden_dim, num_layers, fc_dim, device):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if rnn_model == 'LSTM':
            self.rnn = nn.ModuleList([
                nn.LSTM(embedding_dim if i == 0 else hidden_dim, hidden_dim, num_layers = 1, batch_first = True)
                for i in range(num_layers)
            ])
        else:
            self.rnn = nn.ModuleList([
                nn.RNN(embedding_dim if i == 0 else hidden_dim, hidden_dim, num_layers = 1, batch_first = True, nonlinearity='tanh')
                for i in range(num_layers)
            ])
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn_model = rnn_model

    def forward(self, x, hidden = None, **kwargs):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embedding(x)
        if self.rnn_model == 'LSTM':
            for i, lstm_layer in enumerate(self.rnn):
                x, hidden = checkpoint(lstm_layer, x, hidden)
        else:
            for i, rnn_layer in enumerate(self.rnn):
                x, hidden = rnn_layer(x, hidden[0].unsqueeze(0) if len(hidden[0].shape) == 2 else hidden[0])
                hidden = tuple(hidden,)
        out = self.fc1(x)
        return out, hidden

    def init_hidden(self, batch_size):
            return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

class ParityRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, device):
        super(ParityRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x, hidden = None, **kwargs):
        lengths = kwargs['lengths']
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first = True, enforce_sorted = False)
        packed_output, hidden = self.lstm(packed_input, h0)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lengths = lengths.unsqueeze(1).unsqueeze(2).to(self.device)
        out = torch.gather(out, 1, (lengths - 1).expand(-1, -1, out.shape[-1])).squeeze(1)
        out = self.fc(self.relu(out))
        return out, tuple(hidden,)

if __name__ == '__main__':
     import transformers
     tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', cache_dir = '/storage/vsub851/.cache')
     tokenizer.pad_token = tokenizer.eos_token
     input_ids = tokenizer(['This is a sentence', 'This is another sentence', 'This is a third sentence', 'This is a fourth sentence'], return_tensors = 'pt', padding = 'max_length', max_length = 256)['input_ids']
     print(input_ids.shape)
     model = RNNLM('LSTM', 50257, 256, 512, 3, 512)

     out, hidden = model(input_ids)
     print(out.shape)

     shift_logits = out[..., :-1, :].contiguous()
     shift_labels = input_ids[..., 1:].contiguous()

     print(F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))