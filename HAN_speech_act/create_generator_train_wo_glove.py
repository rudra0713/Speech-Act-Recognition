"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os, nltk, sys
import argparse
import pickle, json
from nltk.tokenize import sent_tokenize, word_tokenize


def get_args():
    parser = argparse.ArgumentParser(
        """Document Classification""")
    parser.add_argument("--train_set", type=str, default="../data/switchboard_train/train_transformer.json")
    args = parser.parse_args()
    return args


def train(opt):

    data_path = opt.train_set
    word_to_idx = {'<unk>': 0}
    with open(data_path) as t_file:
        lines = t_file.readlines()
        for idx, line in enumerate(lines):
            sentences = json.loads(line)['sentences']
            for sentence in sentences:
                for word in sentence:
                    if word.lower() not in word_to_idx:
                        word_to_idx[word.lower()] = len(word_to_idx)
    print("total words ", len(word_to_idx))
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    print("total indices ", len(idx_to_word))

    pickle.dump(word_to_idx, open("word_to_idx.p", "wb"))
    pickle.dump(idx_to_word, open("idx_to_word.p", "wb"))

    return


if __name__ == "__main__":
    opt = get_args()
    train(opt)
