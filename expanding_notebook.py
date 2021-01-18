

import pandas as pd
import numpy as np
from pathlib import Path

import lightgbm as lgb

from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
import seaborn as sns


import lightgbm as lgb
import gc


# plot the feature importance in terms of gain and split
from tensorflow.python.data.experimental.ops.optimization import model


def show_feature_importances(model, features, importance_type):
    max_num_features = 10 ** 10
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = features
    feature_importances['value'] = pd.DataFrame(model.feature_importance(importance_type))
    feature_importances = feature_importances.sort_values(by='value', ascending=False) # sort feature importance
    # feature_importances.to_csv(f'feature_importances_{importance_type}.csv') # write feature importance to csv
    feature_importances = feature_importances[:max_num_features] # only show max_num_features
    
    plt.figure(figsize=(20, 8))
    plt.xlim([0, feature_importances.value.max()*1.1])
    plt.title(f'Feature {importance_type}', fontsize=18);
    sns.barplot(data=feature_importances, x='value', y='feature', palette='rocket');
    for idx, v in enumerate(feature_importances.value):
        plt.text(v, idx, "  {:.2e}".format(v))


###################################################################
#
# load_train_iterated
#

def load_train_iterated(i):

    l_df = []

    for x in range (i):
        filename = dir / f"train_iterated_{x}.pkl"
        assert filename.is_file()
        l_df.append(pd.read_pickle(filename))


    df = pd.concat(l_df, ignore_index = True)

    return df


DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"

dir = Path(DATA_DIR)
assert dir.is_dir()


def get_features_version_2():
    features = ['task_container_id', 'content_id', 'prior_question_elapsed_time',
           'mean_question_accuracy', 'part', 'bundle_id', 'tag1', 'tag2', 'pTime_mean', 'uTime_mean', 'uTime_median',
           'diff_time', 'idle_time', 'uidle_mean', 'uidle_median', 'num_attempts', 'num_correct', 'nIncorrect',
          'mean_user_accuracy', 'hmean_user_content_accuracy',
          'num_correct_3', 'mean_user_accuracy_3', 'hmean_user_content_accuracy_3',
          'num_correct_50', 'mean_user_accuracy_50', 'hmean_user_content_accuracy_50',
          'num_correct_session_max_diff', 'hmean_user_content_accuracy_session_max_diff',
          'num_correct_session_acc_200', 'hmean_user_content_accuracy_session_acc_200',
          'num_correct_session_acc_500', 'hmean_user_content_accuracy_session_acc_500',
          'num_correct_session_acc_1000', 'hmean_user_content_accuracy_session_acc_1000']


    categorical_idx = [features.index('content_id'), features.index('part'),  features.index('bundle_id'), features.index('tag1'),  features.index('tag2')]
    return features, categorical_idx



df = pd.read_pickle(dir / "train_iterated_va.pkl")

# df = df[-25 * 1000 * 1000:]

df = df.assign(dt = df.timestamp / 1000 / 3600 / 24)

features, idx_categorical_feature = get_features_version_2()


print (set(list(df.columns)) - set(features))

assert set(features) - set(list(df.columns)) == set()

va1 = df.va1

df = df.drop(['user_id', 'timestamp', 'va1'], axis = 1).astype(np.float16).fillna(0)

gc.collect()

# STORE NUM TRAIN

train_data = lgb.Dataset(data = df[features][~va1], label = df['y'][~va1], categorical_feature = None)
val_data = lgb.Dataset(data = df[features][va1],  label = df['y'][va1], categorical_feature = None)

del df
gc.collect()


METRICS = ['auc']
NUM_BOOST_ROUND = 300

VERBOSE_EVAL = 1
METRICS = ['auc']

for num_leaves in [200]:

    lgbm_params = {'objective': 'binary', 'metric': METRICS, 'num_leaves': num_leaves, 'max_depth' : 17}

    evals_result = {}

    model = lgb.train(
        params = lgbm_params,
        train_set = train_data,
        valid_sets = [train_data, val_data],
        num_boost_round = NUM_BOOST_ROUND,
        verbose_eval = VERBOSE_EVAL,
        evals_result = evals_result,
        early_stopping_rounds = 20,
        categorical_feature = idx_categorical_feature,
        feature_name = features

    )

del train_data
del val_data
gc.collect()





# save model
model.save_model(str(dir / f'model_400.lgb'))

model_ = lgb.Booster(model_file = str(my_dir / "model_400.lgb"))

show_feature_importances(model, features, 'gain')

show_feature_importances(model, features, 'split')


###################################################################################################3
#
#  PREDICT OOF AND MERGE WITH SAKT
#
#

# Used in model 6_jan_1231.lgb

def get_features():
    features = ['task_container_id', 'content_id',
       'prior_question_elapsed_time', 'mean_question_accuracy', 'part',
       'bundle_id', 'tag1', 'tag2', 'ptime_t_50000', 'utime_t_50000',
       'idle_t_50000', 'ptime_t_200000', 'utime_t_200000', 'idle_t_200000',
       'ptime_t_500000', 'utime_t_500000', 'idle_t_500000', 'ptime_t_10000000',
       'utime_t_10000000', 'idle_t_10000000', 'ptime_p_2', 'utime_p_2',
       'idle_p_2', 'ptime_p_3', 'utime_p_3', 'idle_p_3', 'ptime_p_5',
       'utime_p_5', 'idle_p_5', 'ptime_p_50', 'utime_p_50', 'idle_p_50',
       'ptime_p_200', 'utime_p_200', 'idle_p_200', 'ptime_p_500',
       'utime_p_500', 'idle_p_500', 'ptime_p_800', 'utime_p_800', 'idle_p_800',
       'diff_time', 'idle_time', 'num_attempts', 'nIncorrect', 'num_correct_2',
       'mean_user_accuracy_2', 'hmean_user_content_accuracy_2',
       'num_correct_3', 'mean_user_accuracy_3',
       'hmean_user_content_accuracy_3', 'num_correct_5',
       'mean_user_accuracy_5', 'hmean_user_content_accuracy_5',
       'num_correct_50', 'mean_user_accuracy_50',
       'hmean_user_content_accuracy_50', 'num_correct_200',
       'mean_user_accuracy_200', 'hmean_user_content_accuracy_200',
       'num_correct_500', 'mean_user_accuracy_500',
       'hmean_user_content_accuracy_500', 'num_correct_800',
       'mean_user_accuracy_800', 'hmean_user_content_accuracy_800',
       'num_correct_session_max_diff',
       'hmean_user_content_accuracy_session_max_diff',
       'num_correct_session_acc_200',
       'hmean_user_content_accuracy_session_acc_200',
       'num_correct_session_acc_500',
       'hmean_user_content_accuracy_session_acc_500',
       'num_correct_session_acc_1000',
       'hmean_user_content_accuracy_session_acc_1000',
       'num_correct_session_acc_5000',
       'hmean_user_content_accuracy_session_acc_5000', 'dt']


    categorical_idx = [features.index('content_id'), features.index('part'),  features.index('bundle_id'), features.index('tag1'),  features.index('tag2')]
    return features, categorical_idx


# MERGE HERE


model = lgb.Booster(model_file = str(dir / "6_jan_1231.lgb"))

df = load_train_iterated(11)

import pickle
# y_lgb_va  = pickle.load(open(str(dir / "y_lgb_va.pkl"), "rb" ) )

# y_lgb_va.shape

df_sakt_pred = pd.read_pickle(dir / "df_pred_va.pkl")
df_sakt_pred.shape
df_sakt_pred.columns


df_lgb_pred = pd.read_pickle(dir / "df_val_pred_lgb.pkl")
df_valid_h = pd.read_pickle(dir / "df_valid_header.pkl")

df_lgb_pred.shape
df_valid_h.shape

(df_lgb_pred.index == df_valid_h.index).all()

df_valid_h = df_valid_h.assign(y_lgb = df_lgb_pred.y_lgb)



df_valid_h = df_valid_h.merge(df_sakt_pred, on = ['user_id', 'timestamp', 'content_id'], how = 'left')

m = df_valid_h.answered_correctly.isna()

(~m).sum() == df_sakt_pred.shape[0]

df_merge = df_valid_h[~m].reset_index(drop = True)


from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression


df = df_merge[['user_id', 'y_lgb', 'answered_correctly', 'y', 'timestamp', 'content_id']]

df = df.assign(y_sakt = df.answered_correctly.astype(np.float32))
df = df.assign(y_lgbm = df.y_lgb.astype(np.float32))


df_l = pd.read_pickle(dir / "FE_lecture.pkl")

df_l = df_l.drop('y', axis = 1)


df_merge = df.merge(df_l, on = ['user_id', 'timestamp'])









df_single = df[df.user_id == 1916356099].sort_values(by = 'timestamp')
df_l_single = df_l[df_l.user_id == 1916356099].sort_values(by = 'timestamp')


pd.merge_asof(df_single, df_l_single, on = 'timestamp')


df_single


m_tr = (df.user_id % 7 < 5)
m_tr.sum() / m_tr.shape[0]

df_train = df[m_tr].reset_index(drop = True)
df_valid = df[~m_tr].reset_index(drop = True)

clf = LinearRegression().fit(df_train[['y_sakt', 'y_lgbm']], df_train.y)

clf.coef_[0]
clf.coef_[1]
clf.intercept_




# Predict/score

y_mean = 0.5 * df_valid.y_sakt + 0.5 * df_valid.y_lgbm
roc_auc_score(df_valid.y, y_mean)

y_lr = 0.26990727 * df_valid.y_sakt + 0.76130486 * df_valid.y_lgbm -0.009848833

roc_auc_score(df_valid.y, y_lr)






y_mean = (0.5 * (df.y_lgbm.values + df.y_sakt.values)).astype(np.float16)


y_w_040 = (0.40 * df.y_lgbm.values + 0.60 * df.y_sakt.values).astype(np.float16)
y_w_042 = (0.42 * df.y_lgbm.values + 0.58 * df.y_sakt.values).astype(np.float16)

y_w_045 = (0.45 * df.y_lgbm.values + 0.55 * df.y_sakt.values).astype(np.float16)
y_w_055 = (0.55 * df.y_lgbm.values + 0.45 * df.y_sakt.values).astype(np.float16)
y_w_058 = (0.58 * df.y_lgbm.values + 0.42 * df.y_sakt.values).astype(np.float16)

y_w_060 = (0.60 * df.y_lgbm.values + 0.40 * df.y_sakt.values).astype(np.float16)

y_w_010 = (0.10 * df.y_lgbm.values + 0.9 * df.y_sakt.values).astype(np.float16)
y_w_090 = (0.90 * df.y_lgbm.values + 0.1 * df.y_sakt.values).astype(np.float16)

df = df.assign (y_mean = y_mean, y_w_055 = y_w_055, y_w_045 = y_w_045, y_w_060 = y_w_060, y_w_040 = y_w_040, y_w_042 = y_w_042, y_w_058 = y_w_058, y_w_010 = y_w_010, y_w_090= y_w_090)


def get_meta_features():
    features = ['prior_question_elapsed_time', 'content_id', 'part', 'tag1', 'y_sakt', 'y_lgbm', 'y_mean', 'y_w_040', 'y_w_042', 'y_w_045', 'y_w_055', 'y_w_060', 'y_w_010']

    categorical_idx = [features.index('part'), features.index('content_id'), features.index('tag1')]
    return features, categorical_idx


features, l_cat = get_meta_features()

set(features) - set(df.columns)
assert set(features) - set(df.columns) == set()

y = df.y

X = df[features]

X_train = X[m_tr]
X_valid = X[~m_tr]

y_train = y[m_tr]
y_valid = y[~m_tr]

train_data = lgb.Dataset(data = X_train, label = y_train, categorical_feature = None)
val_data = lgb.Dataset(data = X_valid, label = y_valid, categorical_feature = None)


METRICS = ['auc']
NUM_BOOST_ROUND = 355
VERBOSE_EVAL = 1
METRICS = ['auc']

num_leaves = 113

lgbm_params = {'objective': 'binary', 'metric': METRICS, 'num_leaves': num_leaves}

evals_result = {}

model = lgb.train(
    params = lgbm_params,
    train_set = train_data,
    valid_sets = [train_data, val_data],
    num_boost_round = NUM_BOOST_ROUND,
    verbose_eval = VERBOSE_EVAL,
    evals_result = evals_result,
    early_stopping_rounds = 100,
    categorical_feature = l_cat
)

# 0.780736


y_p = model.predict(X_valid)

roc_auc_score(df_valid.y, y_p)

roc_auc_score(df_valid.y, y_lr)










y_mean = (0.5 * (df.y_lgbm.values + df.y_sakt.values)).astype(np.float16)


y_w_040 = (0.40 * df.y_lgbm.values + 0.60 * df.y_sakt.values).astype(np.float16)
y_w_042 = (0.42 * df.y_lgbm.values + 0.58 * df.y_sakt.values).astype(np.float16)

y_w_045 = (0.45 * df.y_lgbm.values + 0.55 * df.y_sakt.values).astype(np.float16)
y_w_055 = (0.55 * df.y_lgbm.values + 0.45 * df.y_sakt.values).astype(np.float16)
y_w_058 = (0.58 * df.y_lgbm.values + 0.42 * df.y_sakt.values).astype(np.float16)

y_w_060 = (0.60 * df.y_lgbm.values + 0.40 * df.y_sakt.values).astype(np.float16)

y_w_010 = (0.10 * df.y_lgbm.values + 0.9 * df.y_sakt.values).astype(np.float16)
y_w_090 = (0.90 * df.y_lgbm.values + 0.1 * df.y_sakt.values).astype(np.float16)

df = df.assign (y_mean = y_mean, y_hmean = y_hmean, y_w_055 = y_w_055, y_w_045 = y_w_045, y_w_060 = y_w_060, y_w_040 = y_w_040, y_w_042 = y_w_042, y_w_058 = y_w_058, y_w_010 = y_w_010, y_w_090= y_w_090)

roc_auc_score(df.y.values, df.y_lgbm.values)
roc_auc_score(df.y.values, df.y_sakt.values)

roc_auc_score(df.y.values, y_mean)

roc_auc_score(df.y.values, y_w_045)
roc_auc_score(df.y.values, y_w_055)

def get_meta_features():

    features = ['content_id', 'prior_question_elapsed_time', 'num_attempts', 'num_correct', 'num_total', 'mean_user_accuracy',
                'uTime_median', 'bundle_id', 'tag1','y_lgbm', 'y_sakt', 'y_mean', 'y_w_040', 'y_w_042', 'y_w_045', 'y_w_055', 'y_w_060', 'y_w_010']

    categorical_idx = [features.index('content_id'), features.index('bundle_id'), features.index('tag1')]
    return features, categorical_idx


features, l_cat = get_meta_features()

assert set(features) - set(df.columns) == set()

y = df.y

X = df[features]

y = df.y

num_train = 1800000

train_data = lgb.Dataset(data = X[:num_train], label = y[:num_train], categorical_feature = None)
val_data = lgb.Dataset(data = X[num_train:], label = y[num_train:], categorical_feature = None)


METRICS = ['auc']
NUM_BOOST_ROUND = 350

VERBOSE_EVAL = 1
METRICS = ['auc']

num_leaves = 11

lgbm_params = {'objective': 'binary', 'metric': METRICS, 'num_leaves': num_leaves}

evals_result = {}

model = lgb.train(
    params = lgbm_params,
    train_set = train_data,
    valid_sets = [train_data, val_data],
    num_boost_round = NUM_BOOST_ROUND,
    verbose_eval = VERBOSE_EVAL,
    evals_result = evals_result,
    early_stopping_rounds = 25,
    categorical_feature = l_cat
)

# 0.780736


y_p = model.predict(X[num_train:])

roc_auc_score(df.y.values[num_train:], y_p)

model.save_model(str(dir / f'model_meta.lgb'))

meta_model = lgb.Booster(model_file = str(dir / "model_meta.lgb"))

y_p = meta_model.predict(X[num_train:])

X.columns
['content_id', 'prior_question_elapsed_time', 'num_attempts', 'num_correct', 'num_total',
'mean_user_accuracy', 'uTime_median', 'bundle_id', 'tag1', 'y_lgbm', 'y_sakt', 'y_mean',
'y_w_040', 'y_w_042', 'y_w_045', 'y_w_055', 'y_w_060', 'y_w_010']

df_pred

'y_w_040', \
'y_lgbm', \
'y_w_042',\
'y_sakt',\
'y_w_045',\
'y_w_055',\
'y_w_010', \
'y_mean', \
'y_w_060'


{'y_w_040', 'y_lgbm', 'y_w_042', 'y_sakt', 'y_w_045', 'y_w_055', 'y_w_010', 'y_mean', 'y_w_060'}

set (X.columns) - set(df_pred.columns)

y_mean = (0.5 * (df.y_lgbm.values + df.y_sakt.values)).astype(np.float16)


y_w_040 = (0.40 * df.y_lgbm.values + 0.60 * df.y_sakt.values).astype(np.float16)
y_w_042 = (0.42 * df.y_lgbm.values + 0.58 * df.y_sakt.values).astype(np.float16)

y_w_045 = (0.45 * df.y_lgbm.values + 0.55 * df.y_sakt.values).astype(np.float16)
y_w_055 = (0.55 * df.y_lgbm.values + 0.45 * df.y_sakt.values).astype(np.float16)
y_w_058 = (0.58 * df.y_lgbm.values + 0.42 * df.y_sakt.values).astype(np.float16)

y_w_060 = (0.60 * df.y_lgbm.values + 0.40 * df.y_sakt.values).astype(np.float16)

y_w_010 = (0.10 * df.y_lgbm.values + 0.9 * df.y_sakt.values).astype(np.float16)
y_w_090 = (0.90 * df.y_lgbm.values + 0.1 * df.y_sakt.values).astype(np.float16)



