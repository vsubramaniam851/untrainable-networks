import torch
import torch.nn as nn
import math

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model = 768, nhead = 16, num_layers = 4, seq_len = 50, dim_feedforward = 2048, dropout=0.1):
        super(TransformerLM, self).__init__()
        self.tokens_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(d_model, nhead, dropout = dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                                 	nn.ReLU(),
                                                 	nn.Linear(dim_feedforward, d_model)))
            self.layer_norms_1.append(nn.LayerNorm(d_model, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(d_model, eps=1e-12))
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, hidden = None, **kwargs):
        x = x.transpose(0, 1).contiguous()
        positions = torch.arange(len(x), device = x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device = h.device, dtype = h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal = 1)
        for i, (layer_norm_1, attention, layer_norm_2, feed_forward) in enumerate(zip(self.layer_norms_1, self.attentions,
                                                                                        self.layer_norms_2, self.feed_forwards)):

            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        logits = self.lm_head(h)
        logits = logits.permute(1, 0, 2)
        return logits, (torch.zeros(self.d_model), )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model = 768, nhead = 32, num_layers = 4, dim_feedforward = 2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.final_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, x, hidden = None, **kwargs):
        device = x.device
        seq_len = x.shape[1]
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        for layer in self.decoder:
            x = layer(x, x, tgt_mask=mask)
        
        return self.final_layer(x), (torch.zeros(self.d_model), )
    
class TransEnc(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, device, dropout = 0.1):
        super(TransEnc, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, batch_first = True)
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])

        self.output_layer = nn.Linear(d_model, 2)
    
    def forward(self, x, hidden = None, **kwargs):
        mask = (x == 2)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask = mask)

        non_pad_mask = (~mask).unsqueeze(-1)
        x = x * non_pad_mask.float()
        sum_output = x.sum(dim = 1)
        non_pad_count = non_pad_mask.sum(dim = 1)
        x = sum_output / non_pad_count.clamp(min = 1)
        x = self.output_layer(x)
        return x, (torch.zeros(self.d_model), )