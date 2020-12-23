import torch
import torch.nn as nn
import torch.nn.functional as F


def element_wise_mul_wo_add(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return output


class ClaimArticleAttnNet(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        # each encoder state is equivalent to each sentence (acquired after word gru) in article
        # decoder is equivalent to final representation of claim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + (dec_hid_dim * 2), dec_hid_dim * 2)
        self.v = nn.Parameter(torch.rand(dec_hid_dim * 2))

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim * 2]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim * 2]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim * 2]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim * 2, src sent len]

        # v = [dec hid dim * 2]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim * 2]

        ####### Modified
        # attention = torch.bmm(v, energy).squeeze(1)
        #
        # # attention= [batch size, src len]
        #
        # return F.softmax(attention, dim=1)

        attention = F.softmax(torch.bmm(v, energy), dim=2)

        # attention= [batch size, 1, src len]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        mul_encoder_outputs = element_wise_mul_wo_add(encoder_outputs, attention.squeeze(1))
        # print("MUL ENCODER OUTPUTS")
        # print(mul_encoder_outputs.shape)
        return mul_encoder_outputs.permute(1, 0, 2)  # [src sent len, batch size, enc dim * 2]
        # mul_encoder_outputs = [batch size, 1, enc hid dim * 2]

