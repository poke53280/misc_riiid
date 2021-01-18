


# Operation modes.

# SUBMIT PREDICTION ONLINE
# ----------------------------------
# Load model.
# Load user data bank with full train data.

# For each:
#     Retrieve prediction row.
#     Predict and send


# No: Train line creation. 
# No: Get and display prediction score.
# No: Assertions.



# CREATE TRAIN DATA OFFLINE
#
# Create df_iterated from offline train data
# Init user data bank to zero
# 
# For each:
#      Retrieve prediction row.
#      Retrieve target value, combine with predition row. Combine and save.

# No:  Model
# No:  Preditions

# RUN PREDICTION CODE OFFLINE
#
# Load some user_data and a model trained on train data to a certain point
#
#
# For each:
#    Retrieve prediction row.
#    Predict and send
#    Retrieve target value and calculate AUC
# Check offline == online user_data.
# Check validation score. Compare with online submissions.


import riiideducation
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import time
from sklearn.metrics import roc_auc_score
import pickle
import lightgbm as lgb

DATA_DIR = "../input/riiid-test-answer-prediction"
MY_FILE_DIR = "../input/riiid-bundle"

dir = Path(DATA_DIR)
assert dir.is_dir()
l_dir = list (dir.iterdir())

my_dir = Path(MY_FILE_DIR)
assert my_dir.is_dir()
l_dir = list (my_dir.iterdir())





###################################################################
#
# ONLINE PREDICTION
#

# Load resources

q_mean = pickle.load(open(str(my_dir / "q_mean.pkl"), "rb" ) )
g_correct_question = pickle.load(open(str(my_dir / "g_correct_question.pkl"), "rb" ) )
g_correct_answer = pickle.load(open(str(my_dir / "g_correct_answer.pkl"), "rb" ) )

g_qp_time = pickle.load(open(str(dir / "q_ptime.pkl"), "rb" ) )

anID = np.array(list(g_qp_time.keys()))
anTime = np.array(list(g_qp_time.values()))

idx = np.argsort(anID)

g_ptime_anID = anID[idx]
g_ptime_anTime = anTime[idx]



u = pickle.load(open(str(my_dir / "user_bank_train.pkl"), "rb" ) )
u_t = pickle.load(open(str(my_dir / "user_bank_train_t.pkl"), "rb" ) )
model = lgb.Booster(model_file = str(my_dir / "model.lgb"))

g_part = pickle.load(open(str(dir / "g_part.pkl"), "rb" ) )

print(f"Num users in userbank: {u._next_free}")


features, _ = get_features()

env = riiideducation.make_env()
iter_test = env.iter_test()

df_previous = None

(df_current, _) = next(iter_test):

for (df_current, _) in iter_test:

    if df_previous is not None:

        l_previous_answers = eval(df_current.iloc[0].prior_group_responses)
        l_previous_answered_correctly = eval(df_current.iloc[0].prior_group_answers_correct)
        l_previous_answers = np.array(l_previous_answers)
        l_previous_answered_correctly = np.array(l_previous_answered_correctly)
        m_previous_content_type_question = (df_previous.content_type_id == 0)
        df_previous = df_previous.assign(user_answer = l_previous_answers, answered_correctly = l_previous_answered_correctly)
        m_question = df_previous.content_type_id == 0

        store_previous(df_previous[m_question], u, u_t)
    
    df_pred = create_prediction_frame(df_current, u, u_t, q_mean, g_part, g_ptime_anID, g_ptime_anTime)

    y_p = model.predict(df_pred[features])

    df_pred = df_pred.assign(answered_correctly = y_p)
    
    env.predict(df_pred[['row_id', 'answered_correctly']])

    df_previous = df_current


