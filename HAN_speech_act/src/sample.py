import torch

import numpy as np


def element_wise_mul_2(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        # print("feature 1 ", feature_1)
        # print("feature 2 ", feature_2)
        feature = feature_1 * feature_2
        # print("feature ", feature)
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return output


x = torch.arange(1, 9).view(2, 4)
y = torch.LongTensor(4, 2).random_(0, 10)
#
# y = torch.arange(1, 41).view(2, 4, 5)
# # print(x)
# # print(y)
# # z = torch.cat((x, y), dim=0)
# # print(z)
# #
# # print(x)
# # print(x.transpose(1, 0))
# # print(x.permute(1, 0, 2))
# from HAN.src.utils import matrix_mul, element_wise_mul
#
#
# score = torch.arange(1, 9).view(2, 4)   # [  src sent len, batch size]
# # y = torch.LongTensor(1, 3, 5).random_(0, 10)
#
# values = torch.arange(1, 41).view(2, 4, 5) # [ src sent len, batch size, enc hid dim]

# print("v")
# print(values.shape)
# print(values)
# print(".....")
#
# print("score")
# print(score.shape)
# print(score)
# print(".....")
#
#
# z = element_wise_mul(values, score)
# print("z")
# print(z.shape)
# print(z)



print("ATTENTION 2")

# attention = torch.arange(1, 9).view(4, 2)  # [batch size, src sent len]
# enccoder_outputs = torch.arange(1, 41).view(4, 2, 5)   # [batch size, src sent len, enc dim]
# article_encode = torch.arange(1, 21).view(4, 5)   # [batch size, src sent len, enc dim]
#
#
# arrays = [np.random.randn(3, 4) for _ in range(2)]
# x = np.stack(arrays, axis=0)
# print(x)
# print("...................")
# x += 1
#
# print(x)
# acc = 0
# a = [20, 24, 15]
# b = [30, 30, 20]
# for i in range(len(a)):
#     acc += a[i] / b[i]
# print(acc / len(a))
# print(sum(a) / sum(b))
#
#
# mat = [[nan, nan]]
# b = np.isnan(mat)
# for a in b:
#     for x in a:
#         if x:
#             print("nan found")
#             break
#     break

# print("article encode")
# print(article_encode)
# print("....")
# article_max_length_word = 4
# article_max_length_sentences = 2
#
# article_encode = [sentences[:article_max_length_word] for sentences in article_encode]
# print(article_encode)

# f = torch.bmm(attention.unsqueeze(1), enccoder_outputs)
# print("attention")
# print(attention.shape)
# print(attention)
# print(".....")
#
# print("encoder outputs")
# print(enccoder_outputs.shape)
# print(enccoder_outputs)
# print(".....")
#
# print("......")
# print(f)
# print(f.shape)
# print("mul 2..........")
# n = element_wise_mul_2(enccoder_outputs, attention)
# print(n)
# print(n.shape)


xe = torch.randn(3, 2)
print(xe)
#
#
# g = torch.tensor([[nan, nan],        [nan, nan],
#         [nan, nan]])
# h = torch.tensor([[2, 3],        [4, 5],
#         [1, 2]])
# print(isinstance(g[0][0].item(), int))
# print(isinstance(g[0][0].item(), float))
# print(int(g[0][0]))
# print(isinstance(h[0][0].item(), int))
# print(isinstance(h[0][0].item(), float))