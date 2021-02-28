import argparse
import copy
import os
import pickle
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import (AdamW, get_cosine_schedule_with_warmup)

from dataset import RedditDataset
from loss import loss_function
from model import AdvRedditModel
from utils import gr_metrics, pad_collate_reddit

np.set_printoptions(precision=5)


def train_loop(model, dataloader, optimizer, device, dataset_len, scale):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for bi, inputs in enumerate(dataloader):
        optimizer.zero_grad()

        labels, post_features, lens = inputs

        labels = labels.to(device)
        post_features = post_features.to(device)

        output_normal, output_adv = model(post_features, lens, labels)

        _, preds = torch.max(output_normal, 1)

        loss = loss_function(output_normal, output_adv, labels, scale)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, dataloader, device, dataset_len, scale=1):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    fin_conf = []

    for bi, inputs in enumerate(dataloader):
        labels, post_features, lens = inputs

        labels = labels.to(device)
        post_features = post_features.to(device)

        with torch.no_grad():
            output_normal, output_adv = model(post_features, lens, labels)

        _, preds = torch.max(output_normal, 1)

        loss = loss_function(output_normal, None, labels, scale)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_conf.append(output_normal.cpu().detach().numpy())

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets), fin_conf


def main(config):
    pprint(config)

    batch_size = config['batch_size']

    epochs = config['epochs']

    hidden_dim = config['hidden_dim']
    embedding_dim = config['embed_dim']

    num_layers = config['num_layers']
    dropout = config['dropout']
    learning_rate = config['learning_rate']
    scale = config['scale']

    number_of_runs = config['num_runs']

    metrics_dict = {}

    data_dir = config['data_dir']

    epsilon = config['epsilon']

    for i in trange(number_of_runs):
        data_name = os.path.join(data_dir, f'reddit-bert.pkl')

        with open(data_name, 'rb') as f:
            df = pickle.load(f)

        df_train, df_test, _, __ = train_test_split(
            df, df['label'].tolist(), test_size=0.2, stratify=df['label'].tolist())

        train_dataset = RedditDataset(
            df_train.label.values, df_train.enc.values)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=pad_collate_reddit, shuffle=True)

        test_dataset = RedditDataset(df_test.label.values, df_test.enc.values)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=pad_collate_reddit)

        model = AdvRedditModel(
            embedding_dim, hidden_dim, num_layers, dropout, epsilon)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(device)

        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          weight_decay=0.03)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=epochs)

        early_stop_counter = 0
        early_stop_limit = config['early_stop']

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.inf

        for _ in trange(epochs, leave=False):
            loss, accuracy = train_loop(model,
                                        train_dataloader,
                                        optimizer,
                                        device,
                                        len(train_dataset),
                                        scale)

            if scheduler is not None:
                scheduler.step()

            if loss >= best_loss:
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_loss = loss

            if early_stop_counter == early_stop_limit:
                break

        model.load_state_dict(best_model_wts)
        _, _, y_pred, y_true, conf = eval_loop(model,
                                               test_dataloader,
                                               device,
                                               len(test_dataset),
                                               scale)

        m = gr_metrics(y_pred, y_true)

        if 'Precision' in metrics_dict:
            metrics_dict['Precision'].append(m[0])
            metrics_dict['Recall'].append(m[1])
            metrics_dict['FScore'].append(m[2])
            metrics_dict['OE'].append(m[3])
            metrics_dict['all'].append([y_pred, y_true])
        else:
            metrics_dict['Precision'] = [m[0]]
            metrics_dict['Recall'] = [m[1]]
            metrics_dict['FScore'] = [m[2]]
            metrics_dict['OE'] = [m[3]]
            metrics_dict['all'] = [[y_pred, y_true]]

    df = pd.DataFrame(metrics_dict)

    df.to_csv(
        f'{datetime.now().__format__("%d%m%y_%H%M%S")}_df.csv')

    return df['FScore'].median()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batch-size", type=int, default=8,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs")

    parser.add_argument("--num-runs", type=int, default=50,
                        help="number of runs")

    parser.add_argument("--early-stop", type=int, default=10,
                        help="early stop limit")

    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="hidden dimensions")

    parser.add_argument("--embed-dim", type=int, default=768,
                        help="embedding dimensions")

    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of layers")

    parser.add_argument("--dropout", type=float, default=0.4,
                        help="dropout probablity")

    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate")

    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="value of epsilon")

    parser.add_argument("--scale", type=float, default=1.8,
                        help="scale factor alpha")

    parser.add_argument("--data-dir", type=str, default="",
                        help="directory for data")

    args = parser.parse_args()
    main(args.__dict__)
