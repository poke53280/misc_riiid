

###############################################################
#
# rolling_window
#

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


######################################################################3
#
#  create_correct_answer_arrays
#

def create_correct_answer_arrays(df):

    df_correct = df[df.answered_correctly == 1]

    m_duplicated = df_correct.duplicated(subset = 'content_id', keep = 'first')

    df_correct = df_correct[~m_duplicated]

    question = df_correct['content_id'].values
    correct_answer = df_correct['user_answer'].values

    ai_sort = np.argsort(question)

    question = question[ai_sort]
    correct_answer = correct_answer[ai_sort]

    return question, correct_answer


####################################################################
#
#       create_global_time_features
#

def create_global_time_features(df):

    df = df[(df.content_type_id == 0)]

    df = df.reset_index(drop = True)

    df = df.drop('content_type_id', axis = 1)

    assert "content_type_id" not in list(df.columns)

    df_q = pd.read_csv(dir / "questions.csv")

    df_q = df_q.assign(tags = df_q.tags.fillna(""))

    nShape0 = df.shape[0]

    df = df.merge(df_q, left_on = 'content_id', right_on = 'question_id', how = 'inner')

    assert df.shape[0] == nShape0
    m_error = df.tags.isna()

    assert (~m_error).all()


    df = df.sort_values(['user_id', 'timestamp', 'task_container_id'])

    l_dup_subset = ['user_id', 'timestamp', 'task_container_id', 'prior_question_had_explanation', 'prior_question_elapsed_time']

    l_container_columns = l_dup_subset + ['bundle_id']
    

    bundles_df = df[l_container_columns].drop_duplicates(subset = l_dup_subset, keep='first')

    bundles_df["u_ptime"] = bundles_df.groupby('user_id')['prior_question_elapsed_time'].shift(-1).astype(np.float32)

    bundles_df = bundles_df[~bundles_df.u_ptime.isna()]
   

    bundle_ptime_mean = bundles_df[['u_ptime', 'bundle_id']].groupby('bundle_id')['u_ptime'].mean()

    d_q_to_bundle = pd.Series(dict (zip(df_q.question_id, df_q.bundle_id)))


    q_time_mean = d_q_to_bundle.map(bundle_ptime_mean)

    return q_time_mean

