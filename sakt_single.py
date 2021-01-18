
# https://www.kaggle.com/leadbest/sakt-with-randomization-state-updates


# !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import numpy as np
import pandas as pd

import os
from pathlib import Path


MAX_SEQ = 200
N_SKILL = 13523


DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"
MY_FILE_DIR = DATA_DIR

dir = Path(DATA_DIR)
assert dir.is_dir()

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
import gc


#################################################################
#
#       SAKTDataset
#

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):  # HDKIM 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group

        #         self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < 2:  # HDKIM 10
                continue
            self.user_ids.append(user_id)

            # HDKIM Memory reduction
            # if len(q)>self.max_seq:
            #    group[user_id] = (q[-self.max_seq:],qa[-self.max_seq:])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            # HDKIM
            if random.random() > 0.1:
                start = random.randint(0, (seq_len - self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start:end]
                qa[:] = qa_[start:end]
            else:
                # HDKIMHDKIM
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
        else:
            # HDKIM
            if random.random() > 0.1:
                # HDKIMHDKIM
                start = 0
                end = random.randint(2, seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0:seq_len]
                qa[-seq_len:] = qa_[0:seq_len]
            else:
                # HDKIMHDKIM
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label

#################################################################
#
#       FFN
#

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

#################################################################
#
#       future_mask
#

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

#################################################################
#
#       SAKTModel
#

class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim=128):  # HDKIM 100->MAX_SEQ
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill + 1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)
        return x.squeeze(-1), att_weight


#################################################################
#
#       TestDataset
#

class TestDataset(Dataset):
    def __init__(self, samples, test_df, n_skill, max_seq=MAX_SEQ):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.n_skill = n_skill
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]

            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill

        questions = np.append(q[2:], [target_id])

        return x, questions


#################################################################
#
#       train_epoch
#

def train_epoch(model, train_iterator, optim, criterion, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    iter_ = iter(train_iterator)

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()

        optim.zero_grad()
        output, atten_weight = model(x, target_id)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        output = output[:, -1]
        label = label[:, -1]
        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

##############################################################################


# df = create_tidy_train()
df = pd.read_pickle(dir / "train_flat.pkl")


#NUM_META = 2000000  # Cut on df tidy

#num_train = df.shape[0] - NUM_META

group = tidy_train_to_group(df)

group.to_pickle(dir / "group.pkl")




del df
gc.collect()

random.seed(1)

dataset = SAKTDataset(group, N_SKILL)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)  # 512

device = torch.device("cuda")


model = SAKTModel(N_SKILL, embed_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)

epochs = 1
for epoch in range(epochs):
    loss, acc, auc = train_epoch(model, dataloader, optimizer, criterion, device)
    print("epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, loss, acc, auc))


torch.save(model.state_dict(), dir / "sakt_single.pt")
del dataset
gc.collect()


#########################################################################
#
# LOAD AND PREDICT ON TAIL


# df = create_tidy_train()
df = pd.read_pickle(dir / "train_flat.pkl")

num_train = df.shape[0] - NUM_META

group = tidy_train_to_group(df[:num_train])

model_sakt = SAKTModel(N_SKILL, embed_dim=128)

device = torch.device("cuda")

model_sakt.load_state_dict(torch.load(dir / "sakt_single_2000000_it40.pt"))

model_sakt.eval()

model_sakt.to(device)

prev_test_df = None

l_pred = []

df_to_iterate, l_lastAnswerCorrect, l_lastAnswer = create_test_set(df[num_train:])

num_groups = df_to_iterate.index.max() + 1

iter_test = df_iterator(df_to_iterate)

idx = 0

for (test_df, _) in iter_test:
    idx = idx + 1
    if idx % 100 == 0:
        print(idx, num_groups)


    if prev_test_df is not None:
        prev_test_df['answered_correctly'] = (test_df['prior_group_answers_correct'].iloc[0])
        prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
        prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))
        for prev_user_id in prev_group.index:
            prev_group_content = prev_group[prev_user_id][0]
            prev_group_ac = prev_group[prev_user_id][1]
            if prev_user_id in group.index:
                group[prev_user_id] = (np.append(group[prev_user_id][0], prev_group_content),
                                       np.append(group[prev_user_id][1], prev_group_ac))

            else:
                group[prev_user_id] = (prev_group_content, prev_group_ac)
            if len(group[prev_user_id][0]) > MAX_SEQ:
                new_group_content = group[prev_user_id][0][-MAX_SEQ:]
                new_group_ac = group[prev_user_id][1][-MAX_SEQ:]
                group[prev_user_id] = (new_group_content, new_group_ac)

    prev_test_df = test_df.copy()

    test_df = test_df[test_df.content_type_id == False]

    test_dataset = TestDataset(group, test_df, N_SKILL)
    test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)

    outs = []

    for item in test_dataloader:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()

        with torch.no_grad():
            output, att_weight = model_sakt(x, target_id)

        output = torch.sigmoid(output)
        output = output[:, -1]

        outs.extend(output.view(-1).data.cpu().numpy())

    test_df['answered_correctly'] = outs

    l_pred.append(test_df[['user_id', 'timestamp', 'content_id', 'answered_correctly']])

df_pred = pd.concat(l_pred)

df_pred.to_pickle(dir / "sakt_oof.pkl")









