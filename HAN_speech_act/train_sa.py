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
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=256)
    parser.add_argument("--sent_hidden_size", type=int, default=512)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="")
    parser.add_argument("--dev_set", type=str, default="")
    parser.add_argument("--test_set", type=str, default="")
    parser.add_argument("--word_dict_path", type=str, default="word_to_idx.p")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--glove_path", type=str, default="ag_news_csv/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="models")
    parser.add_argument("--model_info", type=str, default="")
    parser.add_argument("--num_sentences", type=int, default=-1)
    parser.add_argument("--left_context_only", type=bool, default=False)

    args = parser.parse_args()
    return args


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    train_gen_len = 0
    for idx, (a_feature, label) in enumerate(iterator):
        train_gen_len += 1
        if torch.cuda.is_available():
            a_feature = a_feature.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        model._init_hidden_state()
        output = model(a_feature)

        loss = criterion(output, label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / train_gen_len


# In[73]:


def evaluate(model, iterator, criterion, testing, total_labels):
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
            if torch.cuda.is_available():
                a_feature = a_feature.cuda()
                label = label.cuda()
            model._init_hidden_state(len(label))
            output = model(a_feature)

            loss = criterion(output, label)

            epoch_loss += loss.item()
            test_examples += len(output)

            if testing:

                for j, ind_output in enumerate(output):
                    max_index = ind_output.max(0)[1]
                    classification.append(max_index.item())
                    ground_truth.append(label[j].item())
                    actual_values[label[j].item()] += 1
                    predicted_values[max_index.item()] += 1

                    if max_index.item() == label[j].item():
                        test_accuracy += 1

    if testing:
        # precision, recall, fscore, support = score(np.array(ground_truth), classification)
        # print("Detailed evaluation:")
        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        # print('f-score: {}'.format(fscore))
        # print('support: {}'.format(support))
        print('test examples: {}'.format(test_examples))
        print('accurate examples: {}'.format(test_accuracy))
        print('accuracy: {}'.format(test_accuracy / test_examples))
        print("actual values ", actual_values)
        print("predicted values ", predicted_values)

    return epoch_loss / len_iterator


def prepare_model(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    # output_file.write("Model's parameters: {}".format(vars(opt)))
    print("number of sentences ", opt.num_sentences)
    print("left context only ", opt.left_context_only)
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

    print("number of classes ", training_set.num_classes)
    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes, word_dict_len, embed_size, opt.num_sentences, opt.left_context_only)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

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
        valid_loss = evaluate(model, all_iters_dev[epoch], criterion, False, training_set.num_classes)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), opt.saved_path + os.sep + opt.model_info + ".pt")

            best_epoch = epoch

        print(
            f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | '
            f'Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

    model.load_state_dict(torch.load(opt.saved_path + os.sep + opt.model_info + ".pt"))

    test_loss = evaluate(model, test_generator, criterion, True, training_set.num_classes)
    print("test result...")
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    print("best epoch ", best_epoch)

    # model.train()
    #
    # num_iter_per_epoch = len(training_generator)
    # for epoch in range(opt.num_epochs):
    #     epoch_train_loss = 0
    #     epoch_dev_loss = 0
    #     ground_truth_train, classification_train = [], []
    #     train_accuracy = 0
    #     train_label_count = 0
    #     ground_truth_dev, classification_dev = [], []
    #     dev_accuracy = 0
    #     dev_label_count = 0
    #
    #     for iter, (a_feature, label) in enumerate(training_generator):
    #
    #         # validate_tensor(c_feature, "error in claim input")
    #         # validate_tensor(a_feature, "error in article input")
    #         if torch.cuda.is_available():
    #             a_feature = a_feature.cuda()
    #             label = label.cuda()
    #         optimizer.zero_grad()
    #         model._init_hidden_state()
    #         predictions = model(a_feature)
    #         train_label_count += len(label)
    #         for j, ind_output in enumerate(predictions):
    #             max_index = ind_output.max(0)[1]
    #             classification_train.append(max_index.item())
    #             ground_truth_train.append(label[j].item())
    #             if max_index.item() == label[j].item():
    #                 train_accuracy += 1
    #         loss = criterion(predictions, label)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_train_loss += loss.item()
    #         # training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
    #         # print("Train Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
    #         #     epoch + 1,
    #         #     opt.num_epochs,
    #         #     iter + 1,
    #         #     num_iter_per_epoch,
    #         #     optimizer.param_groups[0]['lr'],
    #         #     loss, training_metrics["accuracy"]))
    #
    #     epoch_train_loss = epoch_train_loss / len(training_generator)
    #     train_accuracy = train_accuracy / train_label_count
    #     print("epoch ", epoch + 1)
    #     print('Train/Loss', epoch_train_loss)
    #     print('Train/Accuracy', train_accuracy)
    #
    #     writer.add_scalar('Train/Loss', epoch_train_loss)
    #     writer.add_scalar('Train/Accuracy', train_accuracy)
    #
    #     # sys.exit(0)
    #     print("dev begins")
    #     model.eval()
    #     loss_ls = []
    #     dev_label_ls = []
    #     dev_pred_ls = []
    #     for dev_a_feature, dev_label in dev_generator:
    #         num_sample = len(dev_label)
    #         if torch.cuda.is_available():
    #             dev_a_feature = dev_a_feature.cuda()
    #             dev_label = dev_label.cuda()
    #         with torch.no_grad():
    #             # print("a sen len")
    #             # print(a_sen_len)
    #             model._init_hidden_state(num_sample)
    #             dev_predictions = model(dev_a_feature)
    #             dev_loss = criterion(dev_predictions, dev_label)
    #             # loss_ls.append(dev_loss * num_sample)
    #             # dev_label_ls.extend(dev_label.clone().cpu())
    #             # dev_pred_ls.append(dev_predictions.clone().cpu())
    #
    #             dev_label_count += len(dev_label)
    #             epoch_dev_loss += dev_loss.item()
    #
    #             for j, ind_output in enumerate(dev_predictions):
    #                 max_index = ind_output.max(0)[1]
    #                 classification_dev.append(max_index.item())
    #                 ground_truth_dev.append(dev_label[j].item())
    #                 if max_index.item() == dev_label[j].item():
    #                     dev_accuracy += 1
    #
    #     # dev_loss = sum(loss_ls) / dev_set.__len__()
    #     # dev_pred = torch.cat(dev_pred_ls, 0)
    #     # dev_label = np.array(dev_label_ls)
    #     # dev_metrics = get_evaluation(dev_label, dev_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
    #     # output_file.write(
    #     #         "Dev Epoch: {}/{} \nDev loss: {} Dev accuracy: {} \nDev confusion matrix: \n{}\n\n".format(
    #     #             epoch + 1, opt.num_epochs,
    #     #             dev_loss,
    #     #             dev_metrics["accuracy"],
    #     #             dev_metrics["confusion_matrix"]))
    #     #     print("Dev Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
    #     #         epoch + 1,
    #     #         opt.num_epochs,
    #     #         optimizer.param_groups[0]['lr'],
    #     #         dev_loss, dev_metrics["accuracy"]))
    #     epoch_dev_loss = epoch_dev_loss / len(dev_generator)
    #     print("epoch ", epoch + 1)
    #
    #     print('Dev/Loss', epoch_dev_loss)
    #     print('Dev/Accuracy', dev_accuracy / dev_label_count)
    #
    #     writer.add_scalar('Dev/Loss', epoch_dev_loss)
    #     writer.add_scalar('Dev/Accuracy', dev_accuracy)
    #     model.train()
    #
    #     if epoch_dev_loss < best_loss:
    #         best_loss = epoch_dev_loss
    #         best_epoch = epoch
    #         torch.save(model.state_dict(), opt.saved_path + os.sep + "whole_model_han")
    #
    #     # Early stopping
    #     if epoch - best_epoch > opt.es_patience > 0:
    #         print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, dev_loss))
    #
    # # test begins
    # epoch_test_loss = 0
    # ground_truth_test, classification_test = [], []
    # test_accuracy = 0
    # test_label_count = 0
    #
    # model.load_state_dict(torch.load(opt.saved_path + os.sep + "t_model_weight"))
    # model.eval()
    # loss_ls = []
    # te_label_ls = []
    # te_pred_ls = []
    # for te_a_feature, te_label in test_generator:
    #     num_sample = len(te_label)
    #     if torch.cuda.is_available():
    #         te_a_feature = te_a_feature.cuda()
    #         te_label = te_label.cuda()
    #     with torch.no_grad():
    #         model._init_hidden_state(num_sample)
    #         te_predictions = model(te_a_feature)
    #     te_loss = criterion(te_predictions, te_label)
    #     # loss_ls.append(te_loss * num_sample)
    #     # te_label_ls.extend(te_label.clone().cpu())
    #     # te_pred_ls.append(te_predictions.clone().cpu())
    #
    #     test_label_count += len(te_label)
    #     epoch_test_loss += te_loss.item()
    #
    #     for j, ind_output in enumerate(te_predictions):
    #         max_index = ind_output.max(0)[1]
    #         classification_test.append(max_index.item())
    #         ground_truth_test.append(te_label[j].item())
    #         if max_index.item() == te_label[j].item():
    #             test_accuracy += 1
    #             #
    #             # te_loss = sum(loss_ls) / test_set.__len__()
    #             # te_pred = torch.cat(te_pred_ls, 0)
    #             # te_label = np.array(te_label_ls)
    #             # test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
    #             # output_file.write(
    #             #     " Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
    #             #         epoch + 1, opt.num_epochs,
    #             #         te_loss,
    #             #         test_metrics["accuracy"],
    #             #         test_metrics["confusion_matrix"]))
    #             #
    #             # print("Test Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
    #             #     epoch + 1,
    #             #     opt.num_epochs,
    #             #     optimizer.param_groups[0]['lr'],
    #             #     te_loss, test_metrics["accuracy"]))
    #
    # epoch_test_loss = epoch_test_loss / len(test_generator)
    # test_accuracy = test_accuracy / test_label_count
    # print('Test/Loss', epoch_test_loss)
    # print('Test/Accuracy', test_accuracy)
    #
    # writer.add_scalar('Test/Loss', epoch_test_loss)
    # writer.add_scalar('Test/Accuracy', test_accuracy)

    return


if __name__ == "__main__":
    opt = get_args()
    prepare_model(opt)
