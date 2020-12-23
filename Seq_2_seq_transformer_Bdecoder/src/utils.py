"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np, json


# def validate_tensor(a, text):
#     if len(list(a.size())) == 3:
#         for fr in a:
#             for fe in fr:
#                 for v_fe in fe:
#                     try:
#                         x = int(v_fe.item())
#                     except:
#                         print(text, v_fe.item())
#                         print(a)
#
#                         sys.exit()
#     if len(list(a.size())) == 2:
#         for fe in a:
#             for v_fe in fe:
#                 try:
#                     x = int(v_fe.item())
#                 except:
#                     print(text, v_fe.item())
#                     print(a)
#                     sys.exit()
#
#     return
#
#
# def validate_input(a, text):
#     if len(list(a.size())) == 3:
#
#         for fr in a:
#             for fe in fr:
#                 for v_fe in fe:
#                     try:
#                         x = int(v_fe.item())
#                     except:
#                         print(text, v_fe.item())
#                         sys.exit()
#     if len(list(a.size())) == 2:
#         for fe in a:
#             for v_fe in fe:
#                 try:
#                     x = int(v_fe.item())
#                 except:
#                     print(text, v_fe.item())
#                     sys.exit()
#
#     return


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        # if check:
        #     validate_tensor(weight, "found culprit in  weight")
        #     validate_tensor(feature, "found culprit in feature")
            # print("feature ", feature.shape)
            # print("weight shape ", weight.shape)
        feature_m = torch.mm(feature, weight)
        # if check:
        #     # print("///////////////")
        #
        #     validate_tensor(feature_m, "after multiply")

        if isinstance(bias, torch.nn.parameter.Parameter):
            # print("bias shape ", bias.shape)
            # validate_tensor(bias, "before adding bias")

            feature_m = feature_m + bias.expand(feature_m.size()[0], bias.size()[1])
            # validate_tensor(feature_m, "after adding bias")

        feature_m = torch.tanh(feature_m).unsqueeze(0)
        # validate_tensor(feature_m, "after tanh")

        feature_list.append(feature_m)

    return torch.cat(feature_list, 0).squeeze()


def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        # print("feature 1 ", feature_1)
        # print("feature 2 ", feature_2)
        feature = feature_1 * feature_2
        # print("feature ", feature)
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(data_path):
    claim_word_length_list = []
    claim_sent_length_list = []
    article_word_length_list = []
    article_sent_length_list = []

    with open(data_path) as t_file:
        lines = t_file.readlines()
        for idx, line in enumerate(lines):
            # print(line)
            claim = json.loads(line)['claim']
            sent_list = sent_tokenize(claim)
            claim_sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                claim_word_length_list.append(len(word_list))

            article = json.loads(line)['article']
            sent_list = sent_tokenize(article)
            article_sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                article_word_length_list.append(len(word_list))

        claim_sorted_word_length = sorted(claim_word_length_list)
        claim_sorted_sent_length = sorted(claim_sent_length_list)

        article_sorted_word_length = sorted(article_word_length_list)
        article_sorted_sent_length = sorted(article_sent_length_list)

    return claim_sorted_word_length[-1], \
           claim_sorted_sent_length[-1],\
           article_sorted_word_length[int(0.8*len(article_sorted_word_length))], \
           article_sorted_sent_length[int(0.8*len(article_sorted_sent_length))],


    # with open(data_path) as csv_file:
    #     reader = csv.reader(csv_file, quotechar='"')
    #     for idx, line in enumerate(reader):
    #         # print(line)
    #         text = ""
    #         for tx in line[1:]:
    #             text += tx.lower()
    #             text += " "
    #         sent_list = sent_tokenize(text)
    #         sent_length_list.append(len(sent_list))
    #
    #         for sent in sent_list:
    #             word_list = word_tokenize(sent)
    #             word_length_list.append(len(word_list))
    #      sorted_word_length = sorted(word_length_list)
    #     sorted_sent_length = sorted(sent_length_list)
    #
    # return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

def ui():
    print("hello world")


if __name__  ==  "__main__":
    # word, sent = get_max_lengths("../data/test.csv")
    # print (word)
    # print (sent)
    # ui()
    get_max_lengths("../../model/data/cd/dev.json")





