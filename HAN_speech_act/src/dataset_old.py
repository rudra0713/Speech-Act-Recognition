"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import json


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path):
        super(MyDataset, self).__init__()
        articles, labels =  [], []
        with open(data_path) as t_file:
            lines = t_file.readlines()
            for idx, line in enumerate(lines):
                article = json.loads(line)['sentences']
                label = json.loads(line)['label']

                articles.append(article)
                labels.append(label)
                # print(text)
                # print(label)
                # print("...")
        self.articles = articles
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        article = self.articles[index]
        # print("index ", index)
        # print("label ", label)
        # print("article ", article)
        article_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in sentence] for sentence
            in article]

        article_encode = np.stack(arrays=article_encode, axis=0)
        article_encode += 1

        return article_encode.astype(np.int64), label


if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
