


#######################################################################
#
#  create_tidy_train
#

def create_tidy_train():

    df = pd.read_csv(dir / "train.csv")

    l_columns = list(set(list(df.columns)) - set(['row_id']))

    m = df.duplicated(subset = l_columns, keep = 'last')

    df = df[~m].reset_index(drop = True)


    df = df.assign(content_id = df.content_id.astype(np.uint16))
    df = df.assign(content_type_id = df.content_type_id.astype(np.uint8))
    df = df.assign(task_container_id = df.task_container_id.astype(np.uint16))
    df = df.assign(row_id = df.row_id.astype(np.uint32))
    
    df = df.assign(user_answer = df.user_answer.astype(np.int8))
    df = df.assign(user_id = df.user_id.astype(np.uint32))

    prior_question_had_explanation = df.prior_question_had_explanation.fillna(False).astype(np.uint8)

    df = df.assign(prior_question_had_explanation = prior_question_had_explanation)
    
    df = df.assign(answered_correctly = df.answered_correctly.astype(np.int8))
    
    # Create global time
    days_recorded = 100

    # Assume constant join rate for days_recorded days.

    num_users = np.unique(df.user_id).shape[0]

    num_new_users_per_day = num_users / days_recorded

    start_time = np.random.uniform(low = 0, high = days_recorded, size = num_users).astype(int)

    s_start = pd.Series(index = np.unique(df.user_id), data = start_time)

    s_full = df.user_id.map(s_start)

    global_time = df.timestamp + s_full

    global_time = global_time.sort_values()

    df = df.loc[global_time.index].reset_index(drop = True)

    df = df.assign(row_id = np.arange(df.shape[0]))

    return df

#######################################################################
#
#  tidy_train_to_group
#

def tidy_train_to_group(df):
    df_group = df[['timestamp', 'user_id', 'content_id', 'content_type_id', 'answered_correctly']]
    df_group = df_group.assign(user_id = df_group.user_id.astype(np.int32))
    df_group = df_group.assign(content_id = df_group.content_id.astype(np.int16))
    df_group = df_group.assign(content_type_id=df_group.content_type_id.astype(np.int8))

    df_group = df_group[df_group.content_type_id == False]

    df_group = df_group.sort_values(['timestamp'], ascending=True).reset_index(drop = True)

    group = df_group[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))

    return group

#################################################################
#
#       df_iterator
#

class df_iterator:

    def __init__(self, df):
        self.df = df
        self.iNextGroup = 0
        self.num_groups = df.index.max() + 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.iNextGroup < self.num_groups:
            df_out = self.df.loc[[self.iNextGroup]]
            self.iNextGroup = self.iNextGroup + 1
            return (df_out, None)
        else:
            raise StopIteration


#################################################################
#
#       get_answer_list
#

def get_answer_list(df):

    anAnswer = np.array(df.user_answer)
    anGroup = np.array(df.group_num)

    anDiff = np.diff(anGroup)

    m_change = (anDiff != 0)

    iChange = np.where(m_change)[0] + 1

    l = np.split(anAnswer, iChange)

    iFirstIndex = np.insert(iChange, 0, 0)

    s = pd.Series(index = iFirstIndex[1:], data = l[:-1])

    s2 = pd.Series(np.empty((len(df), 0)).tolist())

    s2 = np.empty(len(df))
    s2[:] = np.nan

    s2 = pd.Series(s2)

    s2.loc[s.index] = s

    s2[0] = []

    return s2, l[-1]


#################################################################
#
#       get_answer_correct_list
#

def get_answer_correct_list(df):

    anAnswer = np.array(df.answered_correctly)
    anGroup = np.array(df.group_num)

    anDiff = np.diff(anGroup)

    m_change = (anDiff != 0)

    assert not (~m_change).all(), "get_answer_correct_list: Single group dataframes not implemented"


    iChange = np.where(m_change)[0] + 1

    l = np.split(anAnswer, iChange)

    iFirstIndex = np.insert(iChange, 0, 0)

    s = pd.Series(index = iFirstIndex[1:], data = l[:-1])

    s2 = np.empty(len(df))
    s2[:] = np.nan

    s2 = pd.Series(s2)

    s2.loc[s.index] = s

    s2[0] = []

    return s2, l[-1]


########################################################
#
#  grp_range()
#

def grp_range(a):
    idx = a.cumsum()
    id_arr = np.ones(idx[-1],dtype=int)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return id_arr.cumsum()

########################################################
#
#  create_test_set_experimental()
#

def create_test_set_experimental(df):

    df_user_seq = df[['user_id', 'timestamp']]

    m = df_user_seq.duplicated(keep = 'first')

    df_user_seq = df_user_seq[~m]

    gc.collect()

    vals, count = np.unique(df_user_seq.user_id.values, return_counts=1)

    out = grp_range(count)[np.argsort(df_user_seq.user_id.values, kind = 'stable').argsort()]

    df_user_seq = df_user_seq.assign(group_num=out)

    df = df.merge(df_user_seq, on = ['user_id', 'timestamp'], how = 'left')

    sAnswerCorrect, l_lastAnswerCorrect = get_answer_correct_list(df)
    df = df.assign(prior_group_answers_correct=sAnswerCorrect)
    df = df.drop('answered_correctly', axis=1)

    sAnswer, l_lastAnswer = get_answer_list(df)
    df = df.assign(prior_group_responses=sAnswer)
    df = df.drop('user_answer', axis=1)

    df = df.set_index('group_num')

    return df, l_lastAnswerCorrect, l_lastAnswer


########################################################
#
#  create_test_set()
#

def create_test_set(df):

    df = df.reset_index(drop = True)

    n_group = df.index//333

    num_groups = n_group.max() + 1
    group_size = df.shape[0] / num_groups

    df = df.assign(group_num = n_group)


    sAnswerCorrect, l_lastAnswerCorrect = get_answer_correct_list(df)
    df = df.assign(prior_group_answers_correct = sAnswerCorrect)
    df = df.drop('answered_correctly', axis = 1)

    sAnswer, l_lastAnswer = get_answer_list(df)
    df = df.assign(prior_group_responses = sAnswer)
    df = df.drop('user_answer', axis = 1)

    df = df.set_index('group_num')

    return df, l_lastAnswerCorrect, l_lastAnswer

#################################################################
#
#       save_train_chunk
#

def save_train_chunk(l_train, iCount):
    df_train = pd.concat(l_train)
    df_train.to_pickle(dir / f"train_iterated_lecture_{iCount}.pkl")