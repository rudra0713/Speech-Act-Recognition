"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn


class WordAttNet(nn.Module):
    def __init__(self, hidden_size):
        super(WordAttNet, self).__init__()
        self.embed_size = 300
        self.embed = nn.Linear(self.embed_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden_state):
        embeddings = self.embed(input.float())
        f_output, h_output = self.gru(embeddings.float(), hidden_state)  # feature output and hidden state output
        hidden = torch.cat((h_output[-2, :, :], h_output[-1, :, :]), dim=1)

        return hidden.unsqueeze(0), h_output


if __name__ == "__main__":
    abc = WordAttNet("../ag_news_csv/glove.6B.50d.txt")