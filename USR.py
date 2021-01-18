

# https://www.kaggle.com/its7171/cv-strategy

import pandas as pd
import random
import gc
from pathlib import Path

random.seed(1)

DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"

dir = Path(DATA_DIR)
assert dir.is_dir()

df_it = load_train_iterated(11)
df_it = df_it.sort_values(by = ['user_id', 'timestamp'])

df_it = df_it.reset_index(drop = True)

valid_split1 = df_it.groupby('user_id').tail(5)

train_split1 = df_it[~df_it.row_id.isin(valid_split1.row_id)]

print(f'{train_split1.y.mean():.3f} {valid_split1.y.mean():.3f}')


max_timestamp_u = df_it[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']

MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0,interval)
    return rand_time_stamp



max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.map(rand_time)

df_it = df_it.merge(max_timestamp_u, on='user_id', how='left')
df_it['virtual_time_stamp'] = df_it.timestamp + df_it['rand_time_stamp']

df_it = df_it.sort_values(['virtual_time_stamp', 'row_id']).reset_index(drop=True)

val_size = 2500000


for cv in range(5):
    valid = df_it[-val_size:]
    train = df_it[:-val_size]
    # check new users and new contents
    new_users = len(valid[~valid.user_id.isin(train.user_id)].user_id.unique())
    #valid_question = valid[valid.content_type_id == 0]
    #train_question = train[train.content_type_id == 0]
    new_contents = len(valid_question[~valid_question.content_id.isin(train_question.content_id)].content_id.unique())
    print(f'cv{cv} {train_question.answered_correctly.mean():.3f} {valid_question.answered_correctly.mean():.3f} {new_users} {new_contents}')
    valid.to_pickle(dir / f'cv{cv+1}_valid.pkl')
    train.to_pickle(dir / f'cv{cv+1}_train.pkl')




# df = pd.read_pickle(dir / "train_flat.pkl")



print(f'{train_split1.answered_correctly.mean():.3f} {valid_split1.answered_correctly.mean():.3f}')


####################################################################
33333

import parquet


parquet.open

df_valid = pd.read_parquet(dir / 'cv1_valid.parquet', engine='pyarrow')

df = pd.read_pickle(dir / "train_flat.pkl")

df = df[['user_id', 'timestamp', 'content_id', 'row_id']].merge(df_valid[['user_id', 'timestamp', 'content_id']], on = ['user_id', 'timestamp', 'content_id'], how = 'inner')

l = list (df.row_id)

df = pd.read_pickle(dir / "train_flat.pkl")

m = df.row_id.isin(l)

df = df.assign(va1 = m)

df.to_pickle(dir / "train_flat_va.pkl")

df = pd.read_pickle(dir / "train_flat_va.pkl")



group_va = tidy_train_to_group(df[df.va1])
group_tr = tidy_train_to_group(df[~df.va1])

group_va.to_pickle(dir / "group_va.pkl")
group_tr.to_pickle(dir / "group_tr.pkl")


df = load_train_iterated(11)
df = df[['user_id', 'timestamp', 'content_id', 'row_id']]
gc.collect()

df = df[['user_id', 'timestamp', 'content_id', 'row_id']].merge(df_valid[['user_id', 'timestamp', 'content_id']], on = ['user_id', 'timestamp', 'content_id'], how = 'inner')

l = list (df.row_id)

df = load_train_iterated(11)

m = df.row_id.isin(l)

df = df.assign(va1 = m)

df.to_pickle(dir / "train_iterated_va.pkl")


########################################################### PREDICT SAKT ####################################


df = pd.read_pickle(dir / "train_flat_va.pkl")

va1 = df.va1
df = df.drop('va1', axis = 1)

df_train = df[~va1]
df_valid = df[va1]

df_train.shape
df_valid.shape

group_tr = tidy_train_to_group(df_train)

df_to_iterate, l_lastAnswerCorrect, l_lastAnswer = create_test_set(df_valid)

df_to_iterate.to_pickle(dir / "df_to_iterate_va.pkl")


# group_va = tidy_train_to_group(df[df.va1])


group_va.to_pickle(dir / "group_va.pkl")
group_tr.to_pickle(dir / "group_tr.pkl")