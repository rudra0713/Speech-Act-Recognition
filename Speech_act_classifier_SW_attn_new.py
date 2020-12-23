import numpy as np, re
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as score
import math
import os


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
print(len(LABEL.vocab))
print(LABEL.vocab)
BATCH_SIZE = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), sort=False, batch_size= BATCH_SIZE, device=device)
print(type(train_iterator))



# In[112]:


# class Attention(nn.Module):
#     def __init__(self, query_dim, key_dim, value_dim):
#         super(Attention, self).__init__()
#         self.scale = 1. / math.sqrt(query_dim)

def attention(query, keys, values, query_dim):
    scale = 1. / math.sqrt(query_dim)


    # Query = Hidden
    # hidden dim => (batch size, enc_hid_dim)
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

    query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
    keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
    energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
    energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination


# In[113]:


def attention_2(q, k, v, d_k, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    # print("attention scores shape: ", scores.shape)
    print("scores ... ")
    print(scores.shape)
    print(scores)
    print("......")
    output = torch.matmul(scores, v)
    # print("attention output shape: ", output.shape)
    return output


# In[114]:


class MultiHeadAttention_1(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        
#         self.d_model = d_model
#         self.d_k = d_model // heads
#         self.h = heads
        self.d_k = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        print("q shape 1 ", q.shape)

        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        # print("d_k: ", self.d_k)
# calculate attention using function we will define next
        energy, linear_comb = attention(q, k, v, self.d_k)
        
        output = self.out(linear_comb)
        # print("multi head attention output: ", output.shape)
    
        return output


# In[115]:


class MultiHeadAttention_2(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
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
        print("q shape 1 ", q.shape)
        print("k shape 1 ", k.shape)
        bs = q.size(0)
        q = torch.unsqueeze(q, dim=1)

        k = k.transpose(0,1)
        v = v.transpose(0,1)
        print("q shape 2 ", q.shape)

        print("k shape 2 ", k.shape)


        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        print("q shape 3 ", q.shape)
        print("k shape 3 ", k.shape)


# calculate attention using function we will define next
        scores = attention_2(q, k, v, self.d_k, self.dropout)
        
#         concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()        .view(bs, -1, self.d_model)
        print("multi head attention concatenation :", concat.shape)
        output = self.out(torch.squeeze(concat, dim=1))
        print("multi head attention output: ", output.shape)
        print("multi head attention output: ", output)

        return output


# In[116]:


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, output_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dropout = dropout
        self.attn = MultiHeadAttention_2(2, enc_hid_dim*2, 0.1)

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers= 1, bidirectional=True)
        self.linear = nn.Linear(enc_hid_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input):
        
        # input dim => (src_sent_len, batch_size)
        embed = self.dropout(self.embedding(input))
        
        # embed dim => (src_sent_len, batch_size, emb_dim)
        outputs, hidden = self.rnn(embed)
        print("output dimension : ", outputs.shape)
        print(outputs)
        print("......")
        print("hidden shape : ", hidden.shape)
        print(hidden)
        print("....")

        # print("outputs shape :", outputs.shape)
        # print("hidden shape 1:", hidden.shape)
        # outputs dim => (src_sent_len, batch size, enc_hid_dim * num directions)
        # hidden dim => (number of layers * number of directions, batch size, enc_hid_dim)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)
        # print("hidden shape 2:", hidden.shape)
        # hidden[-2, :, :] reduces 3d to 2d tensor since first dimension is now fixed, so dim = 1 is the last dimension
        # hidden dim => (batch size, enc_hid_dim)


        attn_output = self.attn(hidden, outputs, outputs)
        print("attn output shape : ", attn_output.shape)
        print("attn output  : ", attn_output)

        predictions = self.linear(attn_output)
        print("predictions shape : ", predictions.shape)

        
        return predictions


# In[117]:


#total speech act label => 41

INPUT_DIM = len(SPEECH_ACT.vocab)
OUTPUT_DIM = 41
ENC_EMB_DIM = 256
ENC_HID_DIM = 4
ENC_DROPOUT = 0.5
ATTN_DIM = 500


model = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, OUTPUT_DIM, ENC_DROPOUT).to(device)


optimizer = optim.Adam(model.parameters())

# pad_idx = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss()


# In[118]:


def train(epoch, model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        # print("train iterator ", epoch, i)
        src = batch.s
        trg = batch.l
#         print("src")
#         print(src)
#         print("target")
#         print(trg)
        optimizer.zero_grad()
        
        
        output = model(src)
        break
        #
        # loss = criterion(output, trg)
        #
        # loss.backward()
        #
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #
        # optimizer.step()
        #
        # epoch_loss += loss.item()
        
        
    return epoch_loss / len(iterator)





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
        print("classification ", len(classification))
        precision, recall, fscore, support = score(np.array(ground_truth), classification)
        print("Detailed evaluation:")
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print("accuracy : ", accuracy / sum(support))
 
    return epoch_loss / len(iterator)


# In[120]:


N_EPOCHS = 1
CLIP = 0.001
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'sa_weights_50.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    print("epoch ", epoch)
    train_loss = train(epoch, model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, valid_iterator, criterion, False)
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), MODEL_SAVE_PATH)
#
#     print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
#     print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}  |')


# In[ ]:

#
# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
#
# test_loss = evaluate(model, test_iterator, criterion, True)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
#

