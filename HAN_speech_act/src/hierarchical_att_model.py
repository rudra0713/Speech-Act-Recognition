"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn

from .word_att_model import WordAttNet
from .transformer import Transformer


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, word_dict_len, embed_size, num_sentences, left_context_only):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_sentences = num_sentences
        self.left_context_only = left_context_only
        self.d_model = 512
        self.heads = 8
        self.n_layers = 12

        self.word_att_net = WordAttNet(word_hidden_size, word_dict_len, embed_size)
        self.transformer = Transformer(self.d_model, self.n_layers, self.heads, self.num_sentences)
        self.fc_1 = nn.Linear(self.d_model, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.claim_word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        if torch.cuda.is_available():
            self.claim_word_hidden_state = self.claim_word_hidden_state.cuda()

    def forward(self, c_input):

        c_output_list = []
        c_input = c_input.permute(1, 0, 2)
        for i in c_input:
            c_output, self.claim_word_hidden_state = self.word_att_net(i.permute(1, 0), self.claim_word_hidden_state)
            c_output_list.append(c_output)
        c_output = torch.cat(c_output_list, dim=0)
        # print("c output 1", c_output.shape)

        # print("after concatenation ", c_output.shape)   # [src sent len, batch size,, enc hid dim]
        # transformer
        if self.left_context_only:
            f_output = self.transformer(c_output)[-1]  # no right context
        else:
            f_output = self.transformer(c_output)[self.num_sentences // 2]   # take the middle one, left and right are context
        # print(" f output shape ", f_output.shape)
        output = self.fc_1(f_output)
        return output
