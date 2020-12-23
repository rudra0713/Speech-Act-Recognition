#!/usr/bin/env python
# coding: utf-8

# In[3]:


# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from nltk.translate.bleu_score import sentence_bleu
import spacy

import random
import math
import os
import sys


SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# python -m spacy download en


class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(src)

        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.randn(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        
        # hidden should be => (number of layers * number of directions, batch size, dec_hid_dim))
        # but the author did a squeeze operation in the decoder before returning the last hidden state
        # so hidden dimension becomes => (batch size, dec_hid_dim)
        # another reason to do this would be keep hidden dim similar in both encoder and decoder
        
        # hidden dimension becomes => (batch size, dec_hid_dim)
        # encoder_outputs dimension => (src_sent_len, batch size, enc_hid_dim * num directions)
        
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len,1)
        # hidden new dim = > (batch_size, src_len, dec_hid_dim) 
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs new dim = > (batch_size, src_len, enc_hid_dim * 2) 
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        
        # attn dim = > (batch_size, src_len, dec_hid_dim)
        energy = energy.permute(0, 2, 1)
        
        # we want to compute energy whose dimension is => (batch size, dec_hid_dim, source sent len)
        
        # v dim should be => (batch_size, 1, dec-hid_dim)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        # attention dim (batch_size, 1, src_len)
        # torch.bmm(batch1, batch2, out=None) → Tensor
        # If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, out will be a (b×n×p) tensor
        attention = torch.bmm(v, energy).squeeze(1)
        
        # attention dim (batch_size, src_len)
        return F.softmax(attention, dim=1)


# In[7]:


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.attention = attention
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        # self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim)
        # self.fc = nn.Linear(enc_hid_dim * 2 + emb_dim + dec_hid_dim, output_dim)

        self.embedding = nn.Embedding(output_dim, output_dim)
        self.embedding.weight.data = torch.eye(output_dim)
        self.embedding.weight.requires_grad = False
        self.rnn = nn.GRU(enc_hid_dim * 2 + output_dim, dec_hid_dim)
        self.fc = nn.Linear(enc_hid_dim * 2 + output_dim + dec_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_outputs, hidden):

        # print("input dim in decoder : ", input.shape)
        # print("encoder dim in decoder : ", encoder_outputs.shape)
        # input dim => [batch_size]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
#         print("decoder input 1")
#         print(input)
        input = input.unsqueeze(0)
        # input dim => [1, batch_size]
#         print("decoder input 2")
#         print(input)


        embed = self.dropout(self.embedding(input))
#         print("decoder input 3")
#         print("embed dim in decoder : ", embed.shape)

        # embed dimension => (1, batch_size, emb_dim)
        
        a = self.attention(hidden, encoder_outputs)
        # a dim => (batch_size, src_len)
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        a = a.unsqueeze(1)
        # a dim => (batch_size, 1, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # a dim => (batch_size, 1, src_len)
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        # print("encoder outputs dim in decoder : ", encoder_outputs.shape)
        
        weighted = torch.bmm(a, encoder_outputs)
        # weighted dim => (batch_size, 1, enc_hid_dim * 2)
        weighted = weighted.permute(1, 0, 2)
        # weighted dim => (1, batch_size, enc_hid_dim * 2)
        # print("weighted dim in decoder : ", weighted.shape)
        rnn_input = torch.cat((weighted, embed), dim=2)
        #hidden = [batch size, dec hid dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        # outputs dim => (src_sent_len, batch size, enc_hid_dim * num directions)
        # hidden dim => (number of layers * number of directions, batch size, enc_hid_dim)
        
        # here src_sent_len = number of layers = number of directions = 1 for decoder only
        # sp basically
        # outputs dim => (1, batch size, enc_hid_dim * num directions)
        # hidden dim => (1, batch size, enc_hid_dim)
        
        
        prediction = self.fc(torch.cat((output.squeeze(0), weighted.squeeze(0), embed.squeeze(0)), dim=1))
        return prediction, hidden.squeeze(0)



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        #src = [sent len, batch size]
        #trg = [sent len, batch size]
        # print("src shape in seq2seq : ", src.shape)
        # print("trg shape in seq2seq : ", trg.shape)
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
#         print("trg shape", trg.shape)
        trg_vocab_size = self.decoder.output_dim
#         print("batch size ", batch_size)
#         print("max len ", max_len)
#         print("trg vocab size ", trg_vocab_size)
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
#         print("shape of outputs")
#         print(outputs.shape)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        output = trg[0,:]
#         print("first input")
#         print(output)
        for t in range(1, max_len):
#             print("in LOOP")
            output, hidden = self.decoder(output, encoder_outputs, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
#             print("output in loop")
#             print(output)
#             print(output.max(1))
            top1 = output.max(1)[1] # index of the max value
            output = (trg[t] if teacher_force else top1)
        
        return outputs


