"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn


class WordAttNet(nn.Module):
    def __init__(self, hidden_size, word_dict_len, embed_size):
        super(WordAttNet, self).__init__()

        self.lookup = nn.Embedding(num_embeddings=word_dict_len, embedding_dim=embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden_state):

        # print("word attn model ???")
        # print("input shape : ", input.shape)  # [seq len, batch size]
        output = self.lookup(input)
        # print("word attention model .....")
        # print("output shape : ", output) # [seq len, batch size, emb dim]
        # validate_tensor(hidden_state, "error found hidden state")
        # validate_tensor(input, "error found in input")
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        hidden = torch.cat((h_output[-2, :, :], h_output[-1, :, :]), dim=1)

        return hidden.unsqueeze(0), h_output


if __name__ == "__main__":
    abc = WordAttNet("../ag_news_csv/glove.6B.50d.txt")