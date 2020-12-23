"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
# from HAN.src.utils import matrix_mul, element_wise_mul, validate_tensor
from .utils import matrix_mul, element_wise_mul
import sys


class WordAttNet(nn.Module):
    def __init__(self, hidden_size, word_dict_len, embed_size):
        super(WordAttNet, self).__init__()

        self.lookup = nn.Embedding(num_embeddings=word_dict_len, embedding_dim=embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.word_weight = nn.Parameter(torch.randn(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.randn(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.randn(2 * hidden_size, 1))

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        # print("word attn model ???")
        # print("input shape : ", input.shape)  # [seq len, batch size]
        # print("hidden state shape ", hidden_state.shape)
        print(input)
        sys.exit(0)
        output = self.lookup(input)
        # print("word attention model .....")
        # print("output shape : ", output) # [seq len, batch size, emb dim]
        # validate_tensor(hidden_state, "error found hidden state")
        # validate_tensor(input, "error found in input")

        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        # validate_tensor(f_output, "error found f_output")
        # validate_tensor(h_output, "error found f_output")

        # print("f output  ", f_output)
        # print("matrix mul 1")
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        # validate_tensor(output, "error found matrix mul 1")

        # print("matrix mul 1 ", output)
        # print("matrix mul 2")

        output = matrix_mul(output, self.context_weight, False).permute(1,0)
        # validate_tensor(output, "error found matrix mul 2")

        output = F.softmax(output)
        # print("word attention model f_output shape :", f_output.shape)   # [seq len, batch size, emb dim]
        # print("word attention model output shape :", output.shape)   # [batch size, seq len]

        output = element_wise_mul(f_output,output.permute(1,0))
        # validate_tensor(output, "error found matrix mul 3")
        return output, h_output


if __name__ == "__main__":
    abc = WordAttNet("../ag_news_csv/glove.6B.50d.txt")