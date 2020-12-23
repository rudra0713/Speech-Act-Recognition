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
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

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
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim)
        self.fc = nn.Linear(enc_hid_dim * 2 + emb_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, encoder_outputs, hidden):
        
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
        

        # embed dimension => (1, batch_size, emb_dim)
        
        a = self.attention(hidden, encoder_outputs)
        # a dim => (batch_size, src_len)
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        a = a.unsqueeze(1)
        # a dim => (batch_size, 1, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # a dim => (batch_size, 1, src_len)
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        # weighted dim => (batch_size, 1, enc_hid_dim * 2)
        weighted = weighted.permute(1, 0, 2)
        # weighted dim => (1, batch_size, enc_hid_dim * 2)

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


# In[8]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        #src = [sent len, batch size]
        #trg = [sent len, batch size]
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


# In[9]:



OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())

pad_idx = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


# In[10]:


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        print("train iterator ", i)
        src = batch.src
        trg = batch.trg
#         print("src")
#         print(src)
#         print("target")
#         print(trg)
        optimizer.zero_grad()
        
        output = model(src, trg)
#         print("output after seq2seq")
#         print(output)
#         print(output.shape)
        #trg = [sent len, batch size]
        #output = [sent len, batch size, output dim]
        
        #reshape to:
        #trg = [(sent len - 1) * batch size]
        #output = [(sent len - 1) * batch size, output dim]
        x = output[1:].view(-1, output.shape[2])
        y = trg[1:].view(-1)
#         print("X ", x.shape)
#         print(x[0])
#         print("Y ", y.shape)
#         print(y)
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        
    return epoch_loss / len(iterator)


# In[11]:


def process_final(t):
    sentences = []
    for i in range(len(t[0])):
        tensor_out = t[:, i]
        tensor_out = list(tensor_out.numpy())
        sentence = []
        for j in tensor_out:
            if TRG.vocab.itos[j] == '<pad>':
                break
            if TRG.vocab.itos[j] == '<eos>':
                sentence.append(TRG.vocab.itos[j])
                break
            sentence.append(TRG.vocab.itos[j])
        sentences.append(sentence)
    return sentences


def process_output(output):
    y = []
    for elem in output:
        x = []
        z = []
        for arr in elem:
            values, indices = arr.max(0)
            x.append(indices.item())
            z.append(values.item())
#         print("printing x")
#         print(x)
#         print("printing z")
#         print(z)


        y.append(x)
    final_tensor = torch.tensor(y)
    print("final tensor ")
    print(final_tensor)
    return process_final(final_tensor)


# In[12]:


def evaluate(model, iterator, criterion, testing):
    
    model.eval()
    
    epoch_loss = 0
    bleu_score = 0
    count_pair = 0
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg
            
            output = model(src, trg, 0) #turn off teacher forcing
            if testing:
                output_a = output[1:]
                trg_a = trg[1:]

                sent_out = process_output(output_a)
                sent_trg = process_final(trg_a)
                for o, t in zip(sent_out, sent_trg):
                    reference = [t]
                    candidate = o
                    print("reference ")
                    print(reference)
                    print("candidate ")
                    print(candidate)
                    bleu_score += sentence_bleu(reference, candidate)
                    count_pair += 1
            x = output[1:].view(-1, output.shape[2])
            y = trg[1:].view(-1)
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
        if testing:
            print("count pair ", count_pair)
            print("bleu score ")


            print(bleu_score / count_pair)
    return epoch_loss / len(iterator)


# In[13]:


N_EPOCHS = 10
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_attention_bahadanu.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    print("epoch ", epoch)
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion, False)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}  |')


# In[14]:


model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss = evaluate(model, test_iterator, criterion, True)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

