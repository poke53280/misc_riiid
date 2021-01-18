


import pytz
from datetime import datetime,timezone

now_utc = datetime.now(timezone.utc)

now_utc.astimezone(pytz.timezone("Europe/Oslo"))


local_zone = pytz.timezone("Europe/Oslo")

input_t = datetime.strptime('21.10.2020 02:30', '%d.%m.%Y %H:%M')


input_t = datetime.now()

time_utc = local_zone.localize(input_t).astimezone(pytz.utc)

time_utc.strftime('UTC: %d.%m.%Y %H:%M')

time_utc_pluss = time_utc + timedelta(hours = 1)

# Add one hour
from datetime import timedelta



# Show as Oslo time:

time_utc_pluss.astimezone(local_zone).strftime('LOCAL: %d.%m.%Y %H:%M')




datetime.datetime(year = 1988)










t_now.tzinfo

d.tzinfo.utcoffset(t_now)






########################################## POST PROCESS LECTURE DATA ###########################

def load_train_iterated_lecture(i):

    l_df = []

    for x in range (i):
        filename = dir / f"train_iterated_lecture_{x}.pkl"
        assert filename.is_file()
        l_df.append(pd.read_pickle(filename))

    df = pd.concat(l_df, ignore_index = True)

    return df


df = load_train_iterated_lecture(11)

s = (df.lecture_time / 1000000.0).astype(np.float16).fillna(0)

df = df.assign(lecture_time = s)

df_l = pd.read_csv(dir / "lectures.csv")

df = df.merge(df_l[['tag', 'part', 'lecture_id']], on = 'lecture_id', how = 'left')

df = df.rename(columns = {'tag':'lecture_tag', 'part' : 'lecture_part'})

df = df.assign(lecture_tag = df.lecture_tag.astype(np.float16).fillna(0))
df = df.assign(lecture_part = df.lecture_part.astype(np.float16).fillna(0))

df.to_pickle(dir / "FE_lecture.pkl")


#################################################################################################################
#################################################################################################################


df = load_train_iterated(11)
df_valid = pd.read_parquet(dir / 'cv1_valid.parquet', engine='pyarrow')
df = df[['user_id', 'timestamp', 'content_id', 'row_id']].merge(df_valid[['user_id', 'timestamp', 'content_id']], on = ['user_id', 'timestamp', 'content_id'], how = 'inner')

l = list (df.row_id)

df = load_train_iterated(11)
df = df.assign(dt = df.timestamp / 1000 / 3600 / 24)

m = df.row_id.isin(l)
df = df.assign(va1 = m)

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





features, idx_categorical_feature = get_features()

va1 = df.va1
df = df.drop(['user_id', 'timestamp', 'va1'], axis = 1).astype(np.float16).fillna(0)
print (set(list(df.columns)) - set(features))


assert set(features) - set(list(df.columns)) == set()

gc.collect()


train_data = lgb.Dataset(data = df[features][~va1], label = df['y'][~va1], categorical_feature = None)
val_data = lgb.Dataset(data = df[features][va1],  label = df['y'][va1], categorical_feature = None)

#del df
#gc.collect()


METRICS = ['auc']
NUM_BOOST_ROUND = 350

VERBOSE_EVAL = 1
METRICS = ['auc']


for num_leaves in [400]:

    lgbm_params = {'objective': 'binary', 'metric': METRICS, 'num_leaves': num_leaves, 'max_depth' : 27}

    evals_result = {}

    model = lgb.train(
        params = lgbm_params,
        train_set = train_data,
        valid_sets = [train_data, val_data],
        num_boost_round = NUM_BOOST_ROUND,
        verbose_eval = VERBOSE_EVAL,
        evals_result = evals_result,
        early_stopping_rounds = 30,
        categorical_feature = idx_categorical_feature,
        feature_name = features,
    )



del train_data
del val_data
gc.collect()





def list_most_important_features(model, importance_type, max_num_features):
    feature_importances = pd.DataFrame()
    feature_importances['feature'], _ = get_features()
    feature_importances['value'] = pd.DataFrame(model.feature_importance(importance_type))
    feature_importances = feature_importances.sort_values(by='value', ascending=False) # sort feature importance
    feature_importances = feature_importances[:max_num_features] # only show max_num_features
    return feature_importances


def show_feature_importances(model, importance_type, max_num_features=10 ** 10):
    feature_importances = pd.DataFrame()
    feature_importances['feature'], _ = get_features()
    feature_importances['value'] = pd.DataFrame(model.feature_importance(importance_type))
    feature_importances = feature_importances.sort_values(by='value', ascending=False)  # sort feature importance
    # feature_importances.to_csv(f'feature_importances_{importance_type}.csv') # write feature importance to csv
    feature_importances = feature_importances[:max_num_features]  # only show max_num_features

    plt.figure(figsize=(20, 8))
    plt.xlim([0, feature_importances.value.max() * 1.1])
    plt.title(f'Feature {importance_type}', fontsize=18);
    sns.barplot(data=feature_importances, x='value', y='feature', palette='rocket');
    for idx, v in enumerate(feature_importances.value):
        plt.text(v, idx, "  {:.2e}".format(v))






show_feature_importances(model, 'gain', 35)
plt.show()




list(set(list(list_most_important_features(model, 'gain', 35).feature) + list (list_most_important_features(model, 'split', 35).feature)))





show_feature_importances(model, 'split', 25)
plt.show()




model.save_model(str(dir / '6XXX_jan_1231.lgb'))


bucket



modelfile = "6_jan_1XXX231.lgb"
blob = bucket.blob(modelfile)

source_file_name = dir / modelfile
blob.upload_from_filename(source_file_name)

print("File uploaded to {}.".format(bucket.name))



# OTHER


!pip install pyarrow

!pip install parquet

output_file_name = "cv1_valid.parquet"
blob = bucket.get_blob(output_file_name)
blob.download_to_filename(output_file_name)
print("Downloaded blob {} to {}.".format(blob.name, output_file_name))


# https://www.kaggle.com/ammarnassanalhajali/riiid-lgbm-bagging2-sakt-0-781




import numpy as np
import pandas as pd
import psutil

from collections import defaultdict
import datatable as dt
import lightgbm as lgb
from matplotlib import pyplot as plt
# import riiideducation
import random
from sklearn.metrics import roc_auc_score
import gc
from pathlib import Path

DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"
MY_FILE_DIR = DATA_DIR

dir = Path(DATA_DIR)
assert dir.is_dir()



_ = np.seterr(divide='ignore', invalid='ignore')


target = 'answered_correctly'

def get_train_df():
    data_types_dict = {
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        # 'user_answer': 'int8',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'bool'
    }

    train_df = dt.fread(dir / "train.csv", columns=set(data_types_dict.keys())).to_pandas()

    lectures_df = pd.read_csv(dir / "lectures.csv")

    lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')

    lectures_df = pd.get_dummies(lectures_df, columns=['part', 'type_of'])

    part_lectures_columns = [column for column in lectures_df.columns if column.startswith('part')]

    types_of_lectures_columns = [column for column in lectures_df.columns if column.startswith('type_of_')]

    train_lectures = train_df[train_df.content_type_id == True].merge(lectures_df, left_on='content_id', right_on='lecture_id', how='left')

    user_lecture_stats_part = train_lectures.groupby('user_id', as_index = False)[part_lectures_columns + types_of_lectures_columns].sum()


    lecturedata_types_dict = {
        'user_id': 'int32',
        'part_1': 'int8',
        'part_2': 'int8',
        'part_3': 'int8',
        'part_4': 'int8',
        'part_5': 'int8',
        'part_6': 'int8',
        'part_7': 'int8',
        'type_of_concept': 'int8',
        'type_of_intention': 'int8',
        'type_of_solving_question': 'int8',
        'type_of_starter': 'int8'
    }

    user_lecture_stats_part = user_lecture_stats_part.astype(lecturedata_types_dict)


    for column in user_lecture_stats_part.columns:
        if(column !='user_id'):
            user_lecture_stats_part[column] = (user_lecture_stats_part[column] > 0).astype('int8')

    train_df = pd.merge(train_df, user_lecture_stats_part, on='user_id', how="left", right_index=True)

    for column in user_lecture_stats_part.columns:
        if (column != 'user_id'):
            train_df[column] = train_df[column].fillna(0).astype('int8')

    del (train_lectures)

    cum = train_df.groupby('user_id')['content_type_id'].agg(['cumsum', 'cumcount'])
    train_df['user_lecture_cumsum'] = cum['cumsum']
    train_df['user_lecture_lv'] = cum['cumsum'] / cum['cumcount']


    train_df.user_lecture_lv=train_df.user_lecture_lv.astype('float16')
    train_df.user_lecture_cumsum=train_df.user_lecture_cumsum.astype('int8')
    user_lecture_agg = train_df.groupby('user_id')['content_type_id'].agg(['sum', 'count'])


    train_df['prior_question_had_explanation'].fillna(False, inplace=True)
    train_df = train_df.astype(data_types_dict)
    train_df = train_df[train_df[target] != -1].reset_index(drop=True)
    prior_question_elapsed_time_mean=train_df['prior_question_elapsed_time'].mean()
    train_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace=True)


    max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
    #max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max'])
    max_timestamp_u.columns = ['user_id', 'max_time_stamp']


    train_df['lagtime'] = train_df.groupby('user_id')['timestamp'].shift()
    train_df['lagtime']=train_df['timestamp']-train_df['lagtime']
    train_df['lagtime'].fillna(0, inplace=True)
    train_df.lagtime=train_df.lagtime.astype('int32')
    #train_df.drop(columns=['timestamp'], inplace=True)


    lagtime_agg = train_df.groupby('user_id')['lagtime'].agg(['mean'])
    train_df['lagtime_mean'] = train_df['user_id'].map(lagtime_agg['mean'])
    train_df.lagtime_mean=train_df.lagtime_mean.astype('int32')

    user_prior_question_elapsed_time = train_df[['user_id','prior_question_elapsed_time']].groupby(['user_id']).tail(1)
    #max_timestamp_u = train_df[['user_id','timestamp']].groupby(['user_id']).agg(['max'])
    user_prior_question_elapsed_time.columns = ['user_id', 'prior_question_elapsed_time']


    train_df['delta_prior_question_elapsed_time'] = train_df.groupby('user_id')['prior_question_elapsed_time'].shift()
    train_df['delta_prior_question_elapsed_time']=train_df['prior_question_elapsed_time']-train_df['delta_prior_question_elapsed_time']
    train_df['delta_prior_question_elapsed_time'].fillna(0, inplace=True)


    train_df.delta_prior_question_elapsed_time=train_df.delta_prior_question_elapsed_time.astype('int32')


    train_df['timestamp_int']=(train_df['timestamp']/(1000*3600)).astype('int16')


    train_df['lag'] = train_df.groupby('user_id')[target].shift()

    cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
    train_df['user_correctness'] = cum['cumsum'] / cum['cumcount']
    train_df['user_correct_cumsum'] = cum['cumsum']
    train_df['user_correct_cumcount'] = cum['cumcount']
    train_df.drop(columns=['lag'], inplace=True)

    # train_df['user_correctness'].fillna(1, inplace=True)
    train_df['user_correct_cumsum'].fillna(0, inplace=True)
    #train_df['user_correct_cumcount'].fillna(0, inplace=True)
    train_df.user_correctness=train_df.user_correctness.astype('float16')
    train_df.user_correct_cumcount=train_df.user_correct_cumcount.astype('int16')
    train_df.user_correct_cumsum=train_df.user_correct_cumsum.astype('int16')


    train_df.prior_question_had_explanation=train_df.prior_question_had_explanation.astype('int8')
    train_df['lag'] = train_df.groupby('user_id')['prior_question_had_explanation'].shift()



    cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
    train_df['explanation_mean'] = cum['cumsum'] / cum['cumcount']
    train_df['explanation_cumsum'] = cum['cumsum']
    train_df.drop(columns=['lag'], inplace=True)

    train_df['explanation_mean'].fillna(0, inplace=True)
    train_df['explanation_cumsum'].fillna(0, inplace=True)
    train_df.explanation_mean=train_df.explanation_mean.astype('float16')
    train_df.explanation_cumsum=train_df.explanation_cumsum.astype('int16')

    del cum
    gc.collect()

    train_df["attempt_no"] = 1
    train_df.attempt_no=train_df.attempt_no.astype('int8')
    train_df["attempt_no"] = train_df[["user_id","content_id",'attempt_no']].groupby(["user_id","content_id"])["attempt_no"].cumsum()


    explanation_agg = train_df.groupby('user_id')['prior_question_had_explanation'].agg(['sum', 'count'])
    explanation_agg = explanation_agg.astype('int16')
    #train_df.drop(columns=['prior_question_had_explanation'], inplace=True)

    user_agg = train_df.groupby('user_id')[target].agg(['sum', 'count'])
    content_agg = train_df.groupby('content_id')[target].agg(['sum', 'count','var'])
    task_container_agg = train_df.groupby('task_container_id')[target].agg(['sum', 'count','var'])


    user_agg=user_agg.astype('int16')
    content_agg=content_agg.astype('float32')
    task_container_agg=task_container_agg.astype('float32')


    attempt_no_agg=train_df.groupby(["user_id","content_id"])["attempt_no"].agg(['sum'])
    attempt_no_agg=attempt_no_agg.astype('int8')


    train_df['content_count'] = train_df['content_id'].map(content_agg['count']).astype('int32')
    train_df['content_sum'] = train_df['content_id'].map(content_agg['sum']).astype('int32')
    train_df['content_correctness'] = train_df['content_id'].map(content_agg['sum'] / content_agg['count'])
    train_df.content_correctness=train_df.content_correctness.astype('float16')
    train_df['task_container_sum'] = train_df['task_container_id'].map(task_container_agg['sum']).astype('int32')
    train_df['task_container_std'] = train_df['task_container_id'].map(task_container_agg['var']).astype('float16')
    train_df['task_container_correctness'] = train_df['task_container_id'].map(task_container_agg['sum'] / task_container_agg['count'])
    train_df.task_container_correctness=train_df.task_container_correctness.astype('float16')


    questions_df = pd.read_csv(dir / 'questions.csv', usecols=[0, 1,3,4], dtype={'question_id': 'int16','bundle_id': 'int16', 'part': 'int8','tags': 'str'})

    questions_df['part_bundle_id']=questions_df['part']*100000+questions_df['bundle_id']
    questions_df.part_bundle_id=questions_df.part_bundle_id.astype('int32')
    tag = questions_df["tags"].str.split(" ", n = 10, expand = True)
    tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']


    tag.fillna(0, inplace=True)
    tag = tag.astype('int16')
    questions_df =  pd.concat([questions_df,tag],axis=1).drop(['tags'],axis=1)


    questions_df.rename(columns={'question_id':'content_id'}, inplace=True)

    #
    questions_df['content_correctness'] = questions_df['content_id'].map(content_agg['sum'] / content_agg['count'])
    questions_df.content_correctness=questions_df.content_correctness.astype('float16')
    questions_df['content_correctness_std'] = questions_df['content_id'].map(content_agg['var'])
    questions_df.content_correctness_std=questions_df.content_correctness_std.astype('float16')

    part_agg = questions_df.groupby('part')['content_correctness'].agg(['mean', 'var'])
    questions_df['part_correctness_mean'] = questions_df['part'].map(part_agg['mean'])
    questions_df['part_correctness_std'] = questions_df['part'].map(part_agg['var'])
    questions_df.part_correctness_mean=questions_df.part_correctness_mean.astype('float16')
    questions_df.part_correctness_std=questions_df.part_correctness_std.astype('float16')

    bundle_agg = questions_df.groupby('bundle_id')['content_correctness'].agg(['mean'])
    questions_df['bundle_correctness'] = questions_df['bundle_id'].map(bundle_agg['mean'])
    questions_df.bundle_correctness=questions_df.bundle_correctness.astype('float16')

    tags1_agg = questions_df.groupby('tags1')['content_correctness'].agg(['mean', 'var'])
    questions_df['tags1_correctness_mean'] = questions_df['tags1'].map(tags1_agg['mean'])
    questions_df['tags1_correctness_std'] = questions_df['tags1'].map(tags1_agg['var'])
    questions_df.tags1_correctness_mean=questions_df.tags1_correctness_mean.astype('float16')
    questions_df.tags1_correctness_std=questions_df.tags1_correctness_std.astype('float16')

    questions_df.drop(columns=['content_correctness'], inplace=True)

    train_df = pd.merge(train_df, questions_df, on='content_id', how='left', right_index=True)



    del bundle_agg
    del part_agg
    del tags1_agg
    gc.collect()

    len(train_df)

    train_df['user_correctness'].fillna( 1, inplace=True)
    train_df['attempt_no'].fillna(1, inplace=True)
    #
    train_df.fillna(0, inplace=True)

    return train_df




all_categorical_columns= [
    'user_id',
    'content_id',
    'task_container_id',
    'part',
    'tags1',
    'tags2',
    'tags3',
    'tags4',
    'tags5',
    'tags6',
    'bundle_id',
    'part_bundle_id',
    'prior_question_had_explanation',
    'part_1',
    'part_2',
    'part_3',
    'part_4',
    'part_5',
    'part_6',
    'part_7',
    'type_of_concept',
    'type_of_intention',
    'type_of_solving_question',
    'type_of_starter'
]

train_df = get_train_df()


# 18.34 - 18.41  => 7 mins

a = train_df.user_correctness.values
b = train_df.content_correctness.values

h_mean_accuracy = 2 * (a * b) / (a + b)

train_df["h_mean_accuracy"] = h_mean_accuracy

############ SAKT Part I ###############


features = [
    'h_mean_accuracy',
#   'user_id',
    'timestamp_int',
    'lagtime',
    'lagtime_mean',
   # 'content_id',
   # 'task_container_id',
   #    'user_lecture_cumsum', # X
    'user_lecture_lv',
    'prior_question_elapsed_time',
    'delta_prior_question_elapsed_time',
    'user_correctness',
    'user_correct_cumcount', #X
    'user_correct_cumsum', #X
    'content_correctness',
#    'content_correctness_std',
    'content_count',
    'content_sum', #X
#    'task_container_correctness',
#    'task_container_std',
    'task_container_sum',
    'bundle_correctness',
    'attempt_no',
    'part',
    'part_correctness_mean',
    'part_correctness_std',
    'tags1',
#    'tags1_correctness_mean',
#    'tags1_correctness_std',
    'tags2',
#    'tags3',
#    'tags4',
#    'tags5',
#    'tags6',
    'bundle_id',
    'part_bundle_id',
    'explanation_mean',
    'explanation_cumsum',
    'prior_question_had_explanation',
#    'part_1',
#    'part_2',
#    'part_3',
#    'part_4',
#    'part_5',
#    'part_6',
#    'part_7',
#    'type_of_concept',
#    'type_of_intention',
#    'type_of_solving_question',
#    'type_of_starter'
]


l_unknown_categories = list (set(all_categorical_columns) - set(features))
categorical_columns = [x for x in categorical_columns if x not in l_unknown_categories]
assert set(categorical_columns) - set(features) == set()

train_df[features].dtypes


flag_lgbm=True
clfs = list()
params = {
'num_leaves': 350,
'max_bin':700,
'min_child_weight': 0.03454472573214212,
'feature_fraction': 0.58,
'bagging_fraction': 0.58,
#'min_data_in_leaf': 106,
'objective': 'binary',
'max_depth': -1,
'learning_rate': 0.05,
"boosting_type": "gbdt",
"bagging_seed": 11,
"metric": 'auc',
"verbosity": -1,
'reg_alpha': 0.3899927210061127,
'reg_lambda': 0.6485237330340494,
'random_state': 47
}
# train_df=train_df.reset_index(drop=True)

users = train_df['user_id'].drop_duplicates()  # 去重

users = users.sample(frac=0.025)
users_df = pd.DataFrame()
users_df['user_id'] = users.values

valid_df_newuser = pd.merge(train_df, users_df, on=['user_id'], how='inner', right_index=True)
del users_df
del users
gc.collect()
#
train_df.drop(valid_df_newuser.index, inplace=True)

valid_df = train_df.sample(frac=0.09)
train_df.drop(valid_df.index, inplace=True)

valid_df = valid_df.append(valid_df_newuser)
del valid_df_newuser
gc.collect()
#

print('valid_df length：', len(valid_df))

tr_data = lgb.Dataset(train_df[features], label=train_df[target])
va_data = lgb.Dataset(valid_df[features], label=valid_df[target])

del train_df
del valid_df
gc.collect()

model = lgb.train(
    params,
    tr_data,
    #         train_df[features],
    #         train_df[target],
    num_boost_round=500,
    # valid_sets=[(train_df[features],train_df[target]), (valid_df[features],valid_df[target])],
    valid_sets=[tr_data, va_data],
    early_stopping_rounds=20,
    feature_name=features,
    categorical_feature=categorical_columns,
    verbose_eval=1
)
clfs.append(model)
print('auc:', roc_auc_score(valid_df[target], model.predict(valid_df[features])))
# model.save_model(f'model.txt')
lgb.plot_importance(model, importance_type='split')
# plt.show()

del tr_data
del va_data
gc.collect()
#
# del trains
# del valids
# gc.collect()




##### Inference  #####################################






import torch
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')




