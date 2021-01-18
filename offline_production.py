

import pickle
import gc

from testbed import get_collected_data_for_user

from time import perf_counter

DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"
MY_FILE_DIR = DATA_DIR


from pathlib import Path
dir = Path(DATA_DIR)
assert dir.is_dir()
l_dir = list (dir.iterdir())


############################################################################3
#
#   get_meta_info
#
from sklearn.linear_model import LinearRegression

def get_meta_info():
    df = pd.read_pickle(dir / "meta_2mill.pkl")
    df = df[['content_id', 'y_lgbm', 'y_sakt', 'y']]

    anContentId = np.unique(df.content_id)

    l = []

    for x in anContentId:
        clf = LinearRegression().fit(df[df.content_id == x][['y_sakt', 'y_lgbm']], df[df.content_id == x].y)
        num_content = (df.content_id == x).sum()
        l.append((x, num_content, clf.coef_[0], clf.coef_[1], clf.intercept_))

    df_q = pd.DataFrame(l)
    df_q.columns = ['content_id', 'N', 'coeff_sakt', 'coeff_lgbm', 'intercept']

    df_q = df_q.assign(wReg = df_q.N * .3/ 5000)     # Magic numbers

    df_res = pd.DataFrame({'content_id' : np.arange(13523)})

    df_q = df_res.merge(df_q, on = 'content_id', how = 'left')

    df_q = df_q.assign(N = df_q.N.fillna(0))

    df_q = df_q.assign(coeff_sakt = df_q.coeff_sakt.fillna(0.5))
    df_q = df_q.assign(coeff_lgbm = df_q.coeff_lgbm.fillna(0.5))
    df_q = df_q.assign(intercept = df_q.intercept.fillna(0))
    df_q = df_q.assign(wReg = df_q.wReg.fillna(0))

    return df_q


df_meta = get_meta_info()

df_meta.to_pickle(dir / "question_meta.pkl")

###################################################################
#
# OFFLINE TRAIN SET PRODUCTION
#


df = create_tidy_train()

df.to_pickle(dir / "train_flat.pkl")
df = pd.read_pickle(dir / "train_flat.pkl")


diff_timestamp_t = 500 + 1
ptime_t = 150 + 1
user_answer_t = 3 + 1
content_id_t = 13522 + 1



df_q = pickle.load(open(str(dir / "q_stats.pkl"), "rb"))

df_q = df_q.sort_values(by = 'question_id').reset_index(drop = True)

assert (df_q.question_id == df_q.index).all()

d_bundle = df_q.bundle_id.values
d_a = df_q.correct_answer.values
d_part = df_q.part.values
d_tag1 = df_q.tag1.values
d_tag2 = df_q.tag2.values
d_correct_mean = df_q.correct_mean.values
d_ptime = df_q.p_time.values

### END OFFLINE STATISTICS

iMax = df.shape[0]
# iMax = 5000000

u = UserBank(None, 800, np.uint32)
u_t = UserBank(None, 1, np.uint64)


l_train = []
iSaveCount = 0

df_to_iterate, l_lastAnswerCorrect, l_lastAnswer = create_test_set(df[: iMax])

iter_test = df_iterator(df_to_iterate)

num_groups = df_to_iterate.index.max() + 1
df_previous = None

is_run_checks = False

idx = 0

assert u._next_free == 0
assert (u._next_free == u_t._next_free)

t_acc_create = 0
t_acc_assign = 0
t_acc_qa = 0
t_acc_all = 0

# (df_current, _) = next(iter_test)
for (df_current, _) in iter_test:
    # print (idx)

    if df_previous is not None:

        l_previous_answers = df_current.iloc[0].prior_group_responses
        l_previous_answered_correctly = df_current.iloc[0].prior_group_answers_correct

        l_previous_answers = np.array(l_previous_answers)
        l_previous_answered_correctly = np.array(l_previous_answered_correctly)

        m_previous_content_type_question = (df_previous.content_type_id == 0)

        if is_run_checks:
            assert df_previous.shape[0] == len (l_previous_answers), "df_previous.shape[0] == len (l_previous_answers)"
            assert df_previous.shape[0] == len (l_previous_answered_correctly), "df_previous.shape[0] == len (l_previous_answered_correctly)"
            assert df_pred.shape[0] == len (l_previous_answered_correctly[m_previous_content_type_question.values]), "df_previous.shape[0] == len (l_previous_answered_correctly[m_previous_content_type_question])"

        df_pred = df_pred.assign(y = l_previous_answered_correctly[m_previous_content_type_question.values])

        l_train.append(df_pred)

        df_previous = df_previous.assign(user_answer = l_previous_answers)

        m_question = df_previous.content_type_id == 0

        store_previous(df_previous[m_question], u, u_t)

    df_pred, dt_create, dt_assign, dt_qa, dt_all = create_prediction_frame(df_current, u, u_t)

    t_acc_create = t_acc_create + dt_create
    t_acc_assign = t_acc_assign + dt_assign
    t_acc_qa = t_acc_qa + dt_qa
    t_acc_all = t_acc_all + dt_all

    if idx % 100 == 0 and idx > 0:
        print (t_acc_create, t_acc_assign, t_acc_qa, t_acc_all)

        t_acc_create = 0
        t_acc_assign = 0
        t_acc_qa = 0
        t_acc_all = 0

        print(idx, num_groups, len(l_train))

        if len (l_train) > 30000:
            print (f"Saving train chunk {iSaveCount}...")
            save_train_chunk(l_train, iSaveCount)
            l_train = []

            if iSaveCount == 4:
                print(f"Saving userbanks for meta model. Last row_id is {df_previous.row_id.max()}")
                pickle.dump(u, open(dir / "user_bank_train_4.pkl", "wb"), protocol=4)
                pickle.dump(u_t, open(dir / "user_bank_train_t_4.pkl", "wb"), protocol=4)

            iSaveCount = iSaveCount + 1
            gc.collect()

    df_previous = df_current
    idx = idx + 1
    if idx == 11000:
        break


# Tail end:

assert df_previous.shape[0] == len (l_lastAnswer), "df_previous.shape[0] == len (l_lastAnswer)"
assert df_previous.shape[0] == len (l_lastAnswerCorrect), "df_previous.shape[0] == len (l_lastAnswerCorrect)"

m_previous_content_type_question = (df_previous.content_type_id == 0)

assert df_pred.shape[0] == len (l_lastAnswerCorrect[m_previous_content_type_question.values]), "df_previous.shape[0] == len (l_lastAnswerCorrect[m_previous_content_type_question])"

df_pred = df_pred.assign(y = l_lastAnswerCorrect[m_previous_content_type_question.values])

l_train.append(df_pred)

df_previous = df_previous.assign(user_answer = l_lastAnswer, answered_correctly = l_lastAnswerCorrect)

store_previous(df_previous[m_previous_content_type_question], u, u_t)


if len (l_train) > 0:
    print (f"Saving final train chunk {iSaveCount}...")
    save_train_chunk(l_train, iSaveCount)
    l_train = []
    iSaveCount = iSaveCount + 1
    gc.collect()


pickle.dump(u, open(dir / "user_bank_train.pkl", "wb" ), protocol = 4 )
pickle.dump(u_t, open(dir / "user_bank_train_t.pkl", "wb" ), protocol = 4 )



#######################################################################################################################################
#######################################################################################################################################

# Lecture collection


lectures_df = pd.read_csv(dir / "lectures.csv")

d_tag = dict(zip(lectures_df.lecture_id, lectures_df.tag))
d_part = dict(zip(lectures_df.lecture_id, lectures_df.part))
d_type = dict(zip(lectures_df.lecture_id, lectures_df.type_of))


# Timestamp of last lecture. Part of last lecture. Type of last lecture.

# STORE
#
#
# For each user, store in dictionary (time, lecture_id)
#
# d_lec[user_id] = (time, lecture_id)
#

# RETRIEVE

# On user_id, timestamp
# if user_d in d_lec:
# df_pred['lecture_dt'] = df_pred['timestamp'] - d_lec[time]
# df_pred['lecture_id'] = d_lec[lecture_id]










iMax = df.shape[0]
# iMax = 5000000

d_lec = {}

l_train = []
iSaveCount = 0

df_to_iterate, l_lastAnswerCorrect, l_lastAnswer = create_test_set(df[: iMax])

iter_test = df_iterator(df_to_iterate)

num_groups = df_to_iterate.index.max() + 1
df_previous = None

is_run_checks = False

idx = 0

assert len(d_lec) == 0

t_acc_create = 0
t_acc_assign = 0
t_acc_qa = 0
t_acc_all = 0

 # (df_current, _) = next(iter_test)
for (df_current, _) in iter_test:
    # print (idx)

    if df_previous is not None:

        l_previous_answers = df_current.iloc[0].prior_group_responses
        l_previous_answered_correctly = df_current.iloc[0].prior_group_answers_correct

        l_previous_answers = np.array(l_previous_answers)
        l_previous_answered_correctly = np.array(l_previous_answered_correctly)

        m_previous_content_type_question = (df_previous.content_type_id == 0)

        if is_run_checks:
            assert df_previous.shape[0] == len (l_previous_answers), "df_previous.shape[0] == len (l_previous_answers)"
            assert df_previous.shape[0] == len (l_previous_answered_correctly), "df_previous.shape[0] == len (l_previous_answered_correctly)"
            assert df_pred.shape[0] == len (l_previous_answered_correctly[m_previous_content_type_question.values]), "df_previous.shape[0] == len (l_previous_answered_correctly[m_previous_content_type_question])"

        df_pred = df_pred.assign(y = l_previous_answered_correctly[m_previous_content_type_question.values])

        l_train.append(df_pred)

        df_previous = df_previous.assign(user_answer = l_previous_answers)

        m_lecture = df_previous.content_type_id != 0

        df_previous = df_previous[m_lecture][['user_id', 'timestamp', 'content_id']]

        if df_previous.shape[0] > 0:

            m_duplicated = df_previous['user_id'].duplicated(keep = 'last')
            df_previous = df_previous[~m_duplicated]
            df_previous = df_previous.assign(lecture_data = list(zip(df_previous.timestamp, df_previous.content_id)))
            d_this = dict(zip(df_previous.user_id, df_previous.lecture_data))
            d_lec.update(d_this)

    m_lecture = (df_current.content_type_id != 0)
    df_pred = pd.DataFrame(df_current[['row_id', 'user_id', 'timestamp']])
    df_pred = df_pred[~m_lecture]

    lecture_time = df_pred.user_id.map(lambda x: d_lec[x][0] if x in d_lec else -10000000)
    lecture_id = df_pred.user_id.map(lambda x: d_lec[x][1] if x in d_lec else 0)

    dt_lecture = (df_pred.timestamp - lecture_time) / 1000.0

    df_pred = df_pred.assign(lecture_time = lecture_time, lecture_id = lecture_id)


    t_acc_create = t_acc_create + dt_create
    t_acc_assign = t_acc_assign + dt_assign
    t_acc_qa = t_acc_qa + dt_qa
    t_acc_all = t_acc_all + dt_all

    if idx % 100 == 0 and idx > 0:
        print (t_acc_create, t_acc_assign, t_acc_qa, t_acc_all)

        t_acc_create = 0
        t_acc_assign = 0
        t_acc_qa = 0
        t_acc_all = 0

        print(idx, num_groups, len(l_train))

        if len (l_train) > 30000:
            print (f"Saving train chunk {iSaveCount}...")
            save_train_chunk(l_train, iSaveCount)
            l_train = []

            iSaveCount = iSaveCount + 1
            gc.collect()

    df_previous = df_current
    idx = idx + 1
    #if idx == 11000:
        #break


# Tail end:

assert df_previous.shape[0] == len (l_lastAnswer), "df_previous.shape[0] == len (l_lastAnswer)"
assert df_previous.shape[0] == len (l_lastAnswerCorrect), "df_previous.shape[0] == len (l_lastAnswerCorrect)"

m_previous_content_type_question = (df_previous.content_type_id == 0)

assert df_pred.shape[0] == len (l_lastAnswerCorrect[m_previous_content_type_question.values]), "df_previous.shape[0] == len (l_lastAnswerCorrect[m_previous_content_type_question])"

df_pred = df_pred.assign(y = l_lastAnswerCorrect[m_previous_content_type_question.values])

l_train.append(df_pred)

df_previous = df_previous.assign(user_answer = l_lastAnswer, answered_correctly = l_lastAnswerCorrect)

m_lecture = df_previous.content_type_id != 0

df_previous = df_previous[m_lecture][['user_id', 'timestamp', 'content_id']]

if df_previous.shape[0] > 0:

    m_duplicated = df_previous['user_id'].duplicated(keep = 'last')
    df_previous = df_previous[~m_duplicated]
    df_previous = df_previous.assign(lecture_data = list(zip(df_previous.timestamp, df_previous.content_id)))
    d_this = dict(zip(df_previous.user_id, df_previous.lecture_data))
    d_lec.update(d_this)


if len (l_train) > 0:
    print (f"Saving final train chunk {iSaveCount}...")
    save_train_chunk(l_train, iSaveCount)
    l_train = []
    iSaveCount = iSaveCount + 1
    gc.collect()


# Save lecture



















# VERIFY OUTPUT

df_i = pd.read_pickle(dir / f"train_iterated_0.pkl")


df_i
df_i[df_i.user_id == 1802483544]

df_i.hmean_user_content_accuracy_100.max()



l_cols = [['timestamp', 'content_id', 'prior_question_elapsed_time']]
df_u = df[df.user_id == 2147463192].reset_index(drop = True)

df_u = df_u.assign(prior_timestamp = df_u.timestamp.shift(1).fillna(-1))

df_u = df_u.assign(diff_time = df_u.timestamp - df_u.prior_timestamp)

df_u.iloc[:7]

get_collected_data_for_user(user_id, u, u_t)

u_t.get_data_for_user(2147463192)

prior_question_elapsed_time = np.array([0, 22000.0, 17000.0, 2000.0, 9000.0])

prior_question_elapsed_time.mean()


user_id = 2147464207


df.reset_index().groupby('user_id')['group_num'].size()

df[df.user_id == 2147482888  ]


