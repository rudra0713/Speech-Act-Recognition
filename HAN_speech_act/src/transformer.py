import numpy as np, re, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# In[264]:


# CLAIM = Field()
# ARTICLE = Field()
# LABEL = Field(sequential=False, use_vocab=False)
#
# fields = {'claim': ('c', CLAIM), 'label': ('l', LABEL), 'article': ('a', ARTICLE)}
#
# train_data, valid_data, test_data = TabularDataset.splits(
#     path='data',
#     train='train.json',
#     validation='dev.json',
#     test='test.json',
#     format='json',
#     fields=fields
# )
# print(vars(train_data[0]))
#
# CLAIM.build_vocab(train_data)
# ARTICLE.build_vocab(train_data)
# BATCH_SIZE = 128
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data), sort=False, batch_size=BATCH_SIZE, device=device)
#

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=7):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # add constant to embedding
        # added the following line
        x = x.transpose(0, 1)
        seq_len = x.size(1)
        # print("x shape ", x.shape)
        # print("pe shape ", self.pe.shape)

        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


# In[267]:


def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    # print("attention scores shape: ", scores.shape)
    output = torch.matmul(scores, v)
    # print("attention output shape: ", output.shape)
    return output


# In[268]:


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.8):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        # print("q shape 1 ", q.shape)
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # print("q shape 2 ", q.shape)
        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # print("q shape 3 ", q.shape)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        # print("multi head attention concatenation :", concat.shape)
        output = self.out(concat)
        # print("multi head attention output: ", output.shape)

        return output


# In[269]:


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.8):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# In[270]:


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# In[271]:


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.8):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # print("x shape in encoder layer ", x.shape)
        x2 = self.norm_1(x)
        # print("x2 shape in encoder layer ", x2.shape)

        x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# In[272]:


class Encoder(nn.Module):
    def __init__(self,  d_model, N, heads, num_sentences):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, num_sentences)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)


# In[273]:


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, num_sentences):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, num_sentences)

    def forward(self, src):
        e_outputs = self.encoder(src)
        return e_outputs.permute(1, 0, 2),    # [src sent len,  batch size, hid dim]


# In[274]:


# # d_model = 512
# # heads = 8
# # N = 6
# # N_EPOCHS = 1
# # src_vocab = len(CLAIM.vocab)
# model = Transformer(src_vocab, d_model, N, heads)
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)
# optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#
#
# # In[275]:
#
#
# def train(model, iterator, optimizer):
#     model.train()
#     print("iterator len : ", len(iterator))
#     for i, batch in enumerate(iterator):
#         src_c = batch.c
#         preds = model(src_c)
#         print("prediction Shape ", preds.shape)
#         break
#
#     return -1
#
#
# # In[276]:
#
#
# for epoch in range(N_EPOCHS):
#     print("epoch ", epoch)
#     train_loss = train(model, train_iterator, optim)
#
# # In[10]:
#

