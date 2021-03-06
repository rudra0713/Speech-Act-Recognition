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
from torchnlp.word_to_vector import BPEmb  # doctest: +SKIP
vectors = BPEmb(dim=300)

class MyDataset(Dataset):

    def __init__(self, data_path, word_dict):
        super(MyDataset, self).__init__()
        sentences, labels, labels_for_count_only = [], [], []
        with open(data_path) as t_file:
            lines = t_file.readlines()
            for idx, line in enumerate(lines):
                sentence_col = json.loads(line)['sentences']
                labels_each = json.loads(line)['labels']
                sentences.append(sentence_col)
                labels.append(labels_each)
                labels_for_count_only += labels_each
        self.sentences = sentences
        self.labels = labels
        self.full_dict = word_dict
        # self.claim_max_length_word = c_m_len_word
        # self.article_max_length_sentences = a_m_len_sent
        # self.article_max_length_word = a_m_len_word

        self.num_classes = len(set(labels_for_count_only))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = self.labels[index]
        sentences = self.sentences[index]

        # print("claim ", claim)
        # print("article :", article)
        # print("label : ", label)

        context_encode = [[vectors[word].numpy() for word in sentence] for sentence in sentences]
        # print("context encode")
        # print(len(context_encode))
        # print(len(context_encode[0]))
        # print(len(context_encode[0][0]))

        context_encode = np.stack(arrays=context_encode, axis=0)
        labels = np.stack(arrays=labels, axis=0)

        # print("article encode 1", article_encode)
        # print(article_encode)
        #
        # print("article encode 2")

        # print("article encode 2", article_encode)

        return context_encode.astype(np.int64), labels.astype(np.int64)



if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
