"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch, sys
import torch.nn as nn
from .Seq2Seq_sa import Attention, Encoder, Decoder, Seq2Seq
from .word_att_model_simplified import WordAttNet


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, batch_size, num_classes, num_sentences, left_context_only, device):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.num_sentences = num_sentences
        self.left_context_only = left_context_only
        self.enc_dropout = 0.5
        self.enc_hid_dim = 512
        self.dec_hid_dim = 512
        self.dec_emb_dim = 50
        self.dec_dropout = 0.5

        attn = Attention(self.enc_hid_dim, self.dec_hid_dim)
        enc = Encoder(self.word_hidden_size * 2, self.enc_hid_dim, self.dec_hid_dim, self.enc_dropout)
        dec = Decoder(num_classes, self.dec_emb_dim, self.enc_hid_dim, self.dec_hid_dim, self.dec_dropout, attn)

        self.word_att_net = WordAttNet(word_hidden_size)
        self.seq2seq = Seq2Seq(enc, dec, device).to(device)

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.claim_word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        if torch.cuda.is_available():
            self.claim_word_hidden_state = self.claim_word_hidden_state.cuda()

    def forward(self, c_input, labels):

        c_output_list = []
        # print("c input shape ", c_input.shape)
        c_input = c_input.permute(1, 0, 2, 3)
        for i in c_input:
            # print("claim word hidden state ", self.claim_word_hidden_state.shape)
            c_output, self.claim_word_hidden_state = self.word_att_net(i.permute(1, 0, 2), self.claim_word_hidden_state)
            c_output_list.append(c_output)
        c_output = torch.cat(c_output_list, dim=0)

        # print("after concatenation ", c_output.shape)   # [src sent len, batch size, enc hid dim]
        output = self.seq2seq(c_output, labels)
        return output
