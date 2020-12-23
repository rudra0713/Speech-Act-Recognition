"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from HAN.src.utils import get_max_lengths, get_evaluation, validate_tensor, validate_input
# from HAN.src.dataset import MyDataset
# from HAN.src.hierarchical_att_model import HierAttNet
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet

from tensorboardX import SummaryWriter
import argparse, itertools
import shutil
import numpy as np
import pickle, sys, math


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=256)
    parser.add_argument("--sent_hidden_size", type=int, default=512)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="../data/switchboard_train/train_seq2seq_context_left_2.json")
    parser.add_argument("--dev_set", type=str, default="../data/switchboard_dev/dev_seq2seq_context_left_2.json")
    parser.add_argument("--test_set", type=str, default="../data/switchboard_test/test_seq2seq_context_left_2.json")
    parser.add_argument("--word_dict_path", type=str, default="../HAN_speech_act/word_to_idx.p")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--glove_path", type=str, default="ag_news_csv/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="models")
    parser.add_argument("--model_info", type=str, default="")
    parser.add_argument("--num_sentences", type=int, default=3)
    parser.add_argument("--left_context_only", type=bool, default=False)
    parser.add_argument("--imp_sentence", type=int, default=2)

    args = parser.parse_args()
    return args


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    train_gen_len = 0
    for idx, (a_feature, label) in enumerate(iterator):
        train_gen_len += 1
        label = label.permute(1, 0)
        if torch.cuda.is_available():
            a_feature = a_feature.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        model._init_hidden_state()
        # print("a feature shape in train ", a_feature.shape)
        # print("label shape in train : ", label.shape)

        output = model(a_feature, label)
        # print("output shape in train : ", output.shape)

        # trg = [sent len, batch size]
        # output = [sent len, batch size, output dim]

        # reshape to:
        # trg = [(sent len - 1) * batch size]
        # output = [(sent len - 1) * batch size, output dim]
        x = output.view(-1, output.shape[2])
        y = label.contiguous().view(-1)
        # print("X ", x.shape)
        # print("Y ", y.shape)

        loss = criterion(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / train_gen_len


def evaluate(model, iterator, criterion, total_labels):
    model.eval()
    epoch_loss = 0
    ground_truth, classification = [], []
    test_accuracy = 0
    test_examples = 0
    len_iterator = 0
    actual_values = np.zeros(total_labels)
    predicted_values = np.zeros(total_labels)


    with torch.no_grad():

        for iter, (a_feature, label) in enumerate(iterator):

            len_iterator += 1
            label = label.permute(1, 0)

            if torch.cuda.is_available():
                a_feature = a_feature.cuda()
                label = label.cuda()
            model._init_hidden_state(label.shape[1])
            # print(".............")
            # print("a feature shape ", a_feature.shape)
            # print("label shape ", label.shape)
            output = model(a_feature, label)
            x = output.view(-1, output.shape[2])
            y = label.contiguous().view(-1)

            loss = criterion(x, y)

            epoch_loss += loss.item()
            test_examples += len(output)

    return epoch_loss / len_iterator


def test(model, iterator, criterion, total_labels, imp_sentence):
    model.eval()
    epoch_loss = 0
    ground_truth, classification = [], []
    test_accuracy = 0
    test_examples = 0
    len_iterator = 0
    actual_values = np.zeros(total_labels)
    predicted_values = np.zeros(total_labels)

    with torch.no_grad():

        for iter, (a_feature, label) in enumerate(iterator):

            len_iterator += 1
            label = label.permute(1, 0)

            if torch.cuda.is_available():
                a_feature = a_feature.cuda()
                label = label.cuda()
            model._init_hidden_state(label.shape[1])
            output = model(a_feature, label)

            test_examples += label.shape[1]

            x = output[imp_sentence]
            y = label.contiguous()[imp_sentence]

            for j, ind_output in enumerate(x):
                max_index = ind_output.max(0)[1]
                classification.append(max_index.item())
                ground_truth.append(y[j].item())
                if max_index.item() == y[j].item():
                    test_accuracy += 1
    test_accuracy = test_accuracy / test_examples
    print("test accuracy : ", test_accuracy)
    return test_accuracy



def prepare_model(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    # output_file.write("Model's parameters: {}".format(vars(opt)))
    # print("number of sentences ", opt.num_sentences)
    # print("left context only ", opt.left_context_only)
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    word_to_idx = pickle.load(open(opt.word_dict_path, "rb"))
    word_dict_len = len(word_to_idx)
    embed_size = 512
    CLIP = 2
    # print(opt.num_epochs)
    # print(opt.train_set)
    # print(opt.dev_set)
    # print(opt.test_set)

    training_set = MyDataset(opt.train_set, word_to_idx)
    training_generator = DataLoader(training_set, **training_params)

    dev_set = MyDataset(opt.dev_set, word_to_idx)
    dev_generator = DataLoader(dev_set, **training_params)

    test_set = MyDataset(opt.test_set, word_to_idx)
    test_generator = DataLoader(test_set, **test_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes, word_dict_len, embed_size, opt.num_sentences, opt.left_context_only, device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    best_valid_loss = float('inf')
    best_epoch = 0

    all_iters_train = itertools.tee(training_generator, opt.num_epochs)  # create as many as needed
    all_iters_dev = itertools.tee(dev_generator, opt.num_epochs)  # create as many as needed
    # no need to do that for test, because that will be looped only once

    for epoch in range(opt.num_epochs):
        print("epoch ", epoch + 1)

        train_loss = train(model, all_iters_train[epoch], optimizer, criterion, CLIP)
        valid_loss = evaluate(model, all_iters_dev[epoch], criterion, training_set.num_classes)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), opt.saved_path + os.sep + opt.model_info + ".pt")

            best_epoch = epoch

        print(
            f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | '
            f'Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

    model.load_state_dict(torch.load(opt.saved_path + os.sep + opt.model_info + ".pt"))

    test_acc = test(model, test_generator, criterion, training_set.num_classes, opt.imp_sentence)
    # print("test result...")
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    #
    # print("best epoch ", best_epoch)

    return


if __name__ == "__main__":
    # a = pickle.load(open("../HAN_speech_act/word_to_idx.p", "rb"))
    opt = get_args()
    prepare_model(opt)
