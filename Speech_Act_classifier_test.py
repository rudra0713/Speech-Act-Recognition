
# coding: utf-8

# In[5]:


import numpy as np, re
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator, ReversibleField, TabularDataset, Iterator
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as score
import random
import math
import os


# In[6]:


SPEECH_ACT = Field()
LABEL = Field(sequential=False, use_vocab=False)

fields = {'speech_act': ('s', SPEECH_ACT), 'label': ('l', LABEL)}

train_data, valid_data, test_data = TabularDataset.splits(
                                        path = 'data',
                                        train = 'switchboard_train/train.json',
                                        validation = 'switchboard_dev/dev.json',
                                        test = 'switchboard_test/test.json',
                                        format = 'json',
                                        fields = fields
)

SPEECH_ACT.build_vocab(train_data)
LABEL.build_vocab(train_data)
BATCH_SIZE = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), sort=False, batch_size= BATCH_SIZE, device=device)



class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = Hidden
        # hidden dim => (batch size, enc_hid_dim)
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        print("before multiplication")
        print("query ", query)
        print("keys ", keys)
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        print("energy shape : ", energy.shape)
        print("energy 1 : ", energy)
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        print("energy 2 : ", energy)

        print(".......")
        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        print("linear combination shape : ", linear_combination.shape)
        print(linear_combination)
        print("......")
        return energy, linear_combination


# In[8]:


class Encoder(nn.Module):
    def __init__(self, attention, input_dim, emb_dim, enc_hid_dim, output_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dropout = dropout
        self.attention=attention

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=1, bidirectional=False)
        self.linear = nn.Linear(enc_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input):
        
        # input dim => (src_sent_len, batch_size)
        embed = self.dropout(self.embedding(input))
        
        # embed dim => (src_sent_len, batch_size, emb_dim)
        print("embed shape : ", embed.shape)
        embed = torch.rand(2, 1, 256)
        outputs, hidden = self.rnn(embed)
        print("output dimension : ", outputs.shape)
        print(outputs)
        print("......")
        print("hidden shape : ", hidden.shape)
        print(hidden)
        print("....")
        # outputs dim => (src_sent_len, batch size, enc_hid_dim * num directions)
        # hidden dim => (number of layers * number of directions, batch size, enc_hid_dim)
        # hidden = torch.cat((hidden[-2, : , :], hidden[-1, :, :]), dim = 1)
        # hidden[-2, :, :] reduces 3d to 2d tensor since first dimension is now fixed, so dim = 1 is the last dimension
        # hidden dim => (batch size, enc_hid_dim)

        energy, attn_output = self.attention(hidden[0], outputs, outputs)
        print("attn output shape : ", attn_output.shape)
        predictions = self.linear(attn_output)
        print("predictions shape : ", predictions.shape)
        print()
        
        
        return predictions


# In[9]:


#total speech act label => 41

INPUT_DIM = len(SPEECH_ACT.vocab)
OUTPUT_DIM = 41
ENC_EMB_DIM = 256
ENC_HID_DIM = 4
ENC_DROPOUT = 0.5
ATTN_DIM = 4

attn = Attention(ATTN_DIM, ATTN_DIM, ATTN_DIM)
model = Encoder(attn, INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, OUTPUT_DIM, ENC_DROPOUT).to(device)


optimizer = optim.Adam(model.parameters())

# pad_idx = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss()


# In[10]:


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        # print("train iterator ", i)
        src = batch.s
        trg = batch.l
#         print("src")
#         print(src)
#         print("target")
#         print(trg)
        optimizer.zero_grad()
        
        output = model(src)
        break
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# In[11]:


def evaluate(model, iterator, criterion, testing):
    
    model.eval()
    epoch_loss = 0
    ground_truth, classification = [], []
    accuracy = 0

    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.s
            trg = batch.l
            
            output = model(src) #turn off teacher forcing
            loss = criterion(output, trg)

            epoch_loss += loss.item()
            if testing:
                
                for j, ind_output in enumerate(output):
                    max_index = ind_output.max(0)[1]
                    classification.append(max_index.item())
                    ground_truth.append(trg[j].item())
                    if max_index.item() == trg[j].item():
                        accuracy += 1

                    
    if testing:
        print("trg ", trg.shape)
        print("classification ", len(classification))
        precision, recall, fscore, support = score(np.array(ground_truth), classification)
        print("Detailed evaluation:")
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print("accuracy : ", accuracy / sum(support))

 
    return epoch_loss / len(iterator)


# In[12]:


N_EPOCHS = 1
CLIP = 0.001
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'sa_old_attn.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    print("epoch ", epoch)
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    # valid_loss = evaluate(model, valid_iterator, criterion, False)
    #
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), MODEL_SAVE_PATH)
    #
    # print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
    # print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}  |')


# In[ ]:


# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
#
# test_loss = evaluate(model, test_iterator, criterion, True)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

