
# https://www.kaggle.com/leadbest/sakt-with-randomization-state-updates


# !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import numpy as np
import pandas as pd

import os
from pathlib import Path

import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"
MY_FILE_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"

dir = Path(DATA_DIR)
assert dir.is_dir()
l_dir = list (dir.iterdir())

my_dir = Path(MY_FILE_DIR)
assert my_dir.is_dir()
l_dir = list (my_dir.iterdir())


N_SKILL = 13523

device = torch.device("cpu")

group = pickle.load(open(str(my_dir / "group.pkl"), "rb" ) )

MAX_SEQ = 500
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 256
BATCH_SIZE = 128
DROPOUT = 0.1


class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=100):
        super(SAKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq

        self.user_ids = []
        for i, user_id in enumerate(group.index):
            if (i % 10000 == 0):
                print(f'Processed {i} users')
            content_id, answered_correctly = group[user_id]
            if len(content_id) >= ACCEPTED_USER_CONTENT_SIZE:
                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    last_pos = total_questions // self.max_seq
                    for seq in range(last_pos):
                        index = f"{user_id}_{seq}"
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        self.samples[index] = (content_id[start:end], answered_correctly[start:end])
                    if len(content_id[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                        index = f"{user_id}_{last_pos + 1}"
                        self.user_ids.append(index)
                        self.samples[index] = (content_id[end:], answered_correctly[end:])
                else:
                    index = f'{user_id}'
                    self.user_ids.append(index)
                    self.samples[index] = (content_id, answered_correctly)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        content_id, answered_correctly = self.samples[user_id]
        seq_len = len(content_id)

        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            answered_correctly_seq[:] = answered_correctly[-self.max_seq:]
        else:
            content_id_seq[-seq_len:] = content_id
            answered_correctly_seq[-seq_len:] = answered_correctly

        target_id = content_id_seq[1:]
        label = answered_correctly_seq[1:]

        x = content_id_seq[:-1].copy()
        x += (answered_correctly_seq[:-1] == 1) * self.n_skill

        return x, target_id, label


class FFN(nn.Module):
    def __init__(self, state_size=200, forward_expansion=1, bn_size=MAX_SEQ - 1, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)


FFN()

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

future_mask(5)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads=8, dropout=DROPOUT, forward_expansion=1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, forward_expansion=forward_expansion, dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal_2(x + att_output))
        return x.squeeze(-1), att_weight


class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout=DROPOUT, forward_expansion=1, num_layers=1,
                 heads=8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill + 1, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, forward_expansion=forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(x + pos_x)
        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = self.e_embedding(question_ids)
        e = e.permute(1, 0, 2)
        for layer in self.layers:
            att_mask = future_mask(e.size(0)).to(device)
            x, att_weight = layer(e, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout=DROPOUT, forward_expansion=1, enc_layers=1,
                 heads=8):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, max_seq, embed_dim, dropout, forward_expansion, num_layers=enc_layers)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight


def create_model():
    return SAKTModel(N_SKILL, max_seq=MAX_SEQ, embed_dim=EMBED_SIZE, forward_expansion=1, enc_layers=1, heads=8, dropout=0.1)


class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq=100):
        super(TestDataset, self).__init__()
        self.samples, self.user_ids, self.test_df = samples, [x for x in test_df["user_id"].unique()], test_df
        self.n_skill, self.max_seq = n_skill, max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info['user_id']
        target_id = test_info['content_id']

        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            content_id, answered_correctly = self.samples[user_id]

            seq_len = len(content_id)

            if seq_len >= self.max_seq:
                content_id_seq = content_id[-self.max_seq:]
                answered_correctly_seq = answered_correctly[-self.max_seq:]
            else:
                content_id_seq[-seq_len:] = content_id
                answered_correctly_seq[-seq_len:] = answered_correctly

        x = content_id_seq[1:].copy()
        x += (answered_correctly_seq[1:] == 1) * self.n_skill

        questions = np.append(content_id_seq[2:], [target_id])

        return x, questions


model_sakt = create_model()

if device.type == 'cpu':
    model_sakt.load_state_dict(torch.load(my_dir / "sakt.pth_774", map_location=torch.device('cpu')))
else:
    model_sakt.load_state_dict(torch.load(my_dir / "sakt.pth_774"))


model_sakt.eval()

model_sakt.to(device)


(df_current, _) = next(iter_test)


test_df = df_current[df_current.content_type_id == False]

test_dataset = TestDataset(group, test_df, N_SKILL, max_seq=MAX_SEQ)

test_dataloader = DataLoader(test_dataset, batch_size=len(test_df), shuffle=False)

item = next(iter(test_dataloader))

x = item[0].to(device).long()

target_id = item[1].to(device).long()

with torch.no_grad():
    output, _ = model_sakt(x, target_id)

output = torch.sigmoid(output)
output = output[:, -1]
y_sakt = output.cpu().numpy()


