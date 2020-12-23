"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np, pickle
import json
import sys


class MyDataset(Dataset):

    def __init__(self, data_path, word_dict):
        super(MyDataset, self).__init__()
        sentences, labels = [], []
        with open(data_path) as t_file:
            lines = t_file.readlines()
            for idx, line in enumerate(lines):
                sentence_col = json.loads(line)['sentences']
                label = json.loads(line)['label']
                sentences.append(sentence_col)
                labels.append(label)
        self.sentences = sentences
        self.labels = labels
        self.full_dict = word_dict
        # self.claim_max_length_word = c_m_len_word
        # self.article_max_length_sentences = a_m_len_sent
        # self.article_max_length_word = a_m_len_word

        self.num_classes = len(set(self.labels)) + 1  # count for the unk label introduced for seq2seq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        sentences = self.sentences[index]

        # print("claim ", claim)
        # print("article :", article)
        # print("label : ", label)

        context_encode = [
            [self.full_dict[word.lower()] if word.lower() in self.full_dict else 0 for word in sentence] for sentence
            in sentences]

        context_encode = np.stack(arrays=context_encode, axis=0)

        # print("article encode 1", article_encode)
        # print(article_encode)
        #
        # print("article encode 2")

        # print("article encode 2", article_encode)

        return context_encode.astype(np.int64), label



if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
