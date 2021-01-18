
import pickle
import gc
from skimage.util.shape import view_as_windows as viewW

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a, a[:, :-1]), axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]

#################################################################################################
#
#       get_collected_data_for_user
#

def get_collected_data_for_user(user_id, u, u_t):
    d = u.get_data_for_user(user_id)

    prior_timestamp = u_t.get_data_for_user(user_id)

    diff_timestamp = (d % diff_timestamp_t) * 1000

    d = d // diff_timestamp_t

    ptime = (d % ptime_t) * 1000

    d = d // ptime_t

    data_answers = d % user_answer_t
    data_questions = d // user_answer_t

    df_c = pd.DataFrame({'ptime': ptime, 'difftime' : diff_timestamp, 'question' : data_questions, 'answer': data_answers})

    return df_c, prior_timestamp



#################################################################################################
#
#       get_last_n_mask
#

def get_last_n_mask(u, N, iLoc):

    num_rows = iLoc.shape[0]
    offset = (u._anNext[iLoc] + u._width - N) % u._width

    anAdd = np.tile(np.arange(0, N), num_rows).reshape(-1, N)

    anOffset = np.repeat(offset, N).reshape(-1, N)

    anOffset = anOffset + anAdd

    anOffset = anOffset % u._width

    exp_mask = np.zeros(shape = (num_rows, u._width), dtype = np.bool)

    Y = np.repeat(np.arange(num_rows), N).ravel()

    anOffset = anOffset.ravel()

    exp_mask[Y, anOffset.ravel()] = True

    return exp_mask


#################################################################################################
#
#       FeatureBank
#

l_qa_features = []

L_SAMPLING_POSITIONS = [2, 3, 5, 50, 100, 200, 300, 400, 500, 600, 700, 800]
L_T_ACC_MAX = [10000, 20000, 50000, 100000, 200000, 500000, 10000000]

for t_acc_max in L_T_ACC_MAX:
    l_qa_features.append(f"ptime_t_{t_acc_max}")
    l_qa_features.append(f"utime_t_{t_acc_max}")
    l_qa_features.append(f"idle_t_{t_acc_max}")

for i in L_SAMPLING_POSITIONS:
    l_qa_features.append(f"ptime_p_{i}")
    l_qa_features.append(f"utime_p_{i}")
    l_qa_features.append(f"idle_p_{i}")


l_qa_features.append("diff_time")
l_qa_features.append("idle_time")
l_qa_features.append("num_attempts")
l_qa_features.append("nIncorrect")

for i in L_SAMPLING_POSITIONS:
    l_qa_features.append(f"num_correct_{i}")
    l_qa_features.append(f"mean_user_accuracy_{i}")
    l_qa_features.append(f"hmean_user_content_accuracy_{i}")

l_qa_features.append("num_correct_session_max_diff")
l_qa_features.append("hmean_user_content_accuracy_session_max_diff")

l_qa_features.append("num_correct_session_acc_200")
l_qa_features.append("hmean_user_content_accuracy_session_acc_200")

l_qa_features.append("num_correct_session_acc_500")
l_qa_features.append("hmean_user_content_accuracy_session_acc_500")

l_qa_features.append("num_correct_session_acc_1000")
l_qa_features.append("hmean_user_content_accuracy_session_acc_1000")

l_qa_features.append("num_correct_session_acc_5000")
l_qa_features.append("hmean_user_content_accuracy_session_acc_5000")


class FeatureBank:
    def __init__(self):
        self._l_feature = []
        self._l_data = []

    def add(self, zFeature, data):
        assert zFeature not in self._l_feature
        self._l_feature.append(zFeature)
        self._l_data.append(data)

    def get_data_frame(self):

        assert self._l_feature == l_qa_features

        data = np.stack(self._l_data).T
        return pd.DataFrame(data, columns=self._l_feature)

    def get_zero_frame(self, num, num_features):
        data = np.zeros((num, num_features), dtype = np.float16)
        return pd.DataFrame(data, columns=l_qa_features)



#################################################################################################
#
#       session_specific
#

def session_specific(m_valid_data_session, valid_idx_session, zKey, f, data_questions, data_answers, mean_question_accuracy, m_no_data):

    nCount_session = m_valid_data_session.sum(axis=1)

    nCount_session[m_no_data] = 0

    m_correct_session = np.zeros(shape=m_valid_data_session.shape, dtype=np.bool)
    m_correct_session[valid_idx_session] = (data_answers[valid_idx_session] == d_a[data_questions[valid_idx_session]])

    num_correct_session = m_correct_session.sum(axis=1)
    num_correct_session[m_no_data] = 0
    f.add(f"num_correct_session_{zKey}", num_correct_session.ravel().astype(np.float16))

    #nIncorrect_session = nCount_session - num_correct_session
    #nIncorrect_session[m_no_data] = 0
    #f.add(f"nIncorrect_session_{zKey}", nIncorrect.ravel().astype(np.float16))

    f_mean_user_accuracy, hmean_user_content_accuracy = get_accuracy_features(mean_question_accuracy, nCount_session, num_correct_session)
    # f.add(f"mean_user_accuracy_session_{zKey}", f_mean_user_accuracy.ravel().astype(np.float16))
    f.add(f"hmean_user_content_accuracy_session_{zKey}", hmean_user_content_accuracy.ravel().astype(np.float16))

#################################################################################################
#
#       get_qa_features
#

def get_qa_features(df_pred, u, u_t):
    user_id_current = df_pred.user_id.values
    content_id_current = df_pred.content_id.values
    timestamp_current = df_pred.timestamp.values
    ptime_current = df_pred.prior_question_elapsed_time.values
    mean_question_accuracy = df_pred.mean_question_accuracy.values

    num = user_id_current.shape[0]
    f = FeatureBank()

    num_features = len (l_qa_features)

    iLoc = u.get_users(user_id_current)

    m_no_data = iLoc < 0

    if m_no_data.all():
        return f.get_zero_frame(num, num_features)

    # Set to any user data (at 0). Data is invalidated later.
    iLoc[m_no_data] = 0

    d = u._anData[iLoc].copy()
    d = strided_indexing_roll(d, u._width - u._anNext[iLoc])
    d = np.flip(d, axis=1)

    m_VALID_DATA = (d < np.iinfo(u._datatype).max)
    VALID_IDX = np.where(m_VALID_DATA)

    # Unpack
    prior_timestamp = u_t._anData[iLoc]
    diff_timestamp = (d % diff_timestamp_t) * 1000
    d = d // diff_timestamp_t
    ptime = (d % ptime_t) * 1000
    d = d // ptime_t
    data_answers = d % user_answer_t
    data_questions = d // user_answer_t


    ######################################### SESSION - MEANS ###################################

    # p_time global session prepration
    p_time = data_questions.astype(np.float32).copy()
    p_time[VALID_IDX] = d_ptime[data_questions[VALID_IDX]]

    idle_time = diff_timestamp.astype(np.float32) - ptime.astype(np.float32)



    for t_acc_max in L_T_ACC_MAX:

        # Session preparation
        m = diff_timestamp.cumsum(axis=1) < t_acc_max
        m_valid_data_session = m_VALID_DATA & m
        valid_idx_session = np.where(m_valid_data_session)
        session_count = m_valid_data_session.sum(axis=1)

        m_none_in_session = (session_count == 0)
        no_session_data = ( m_no_data | m_none_in_session)

        session_count[no_session_data]  = 1

        # p_time:
        p_time_session = p_time.copy()
        p_time_session[~m_valid_data_session] = 0
        p_time_acc = p_time_session.sum(axis=1)

        session_mean = p_time_acc / session_count
        session_mean[no_session_data] = 0
        f.add(f"ptime_t_{t_acc_max}", (session_mean.ravel()/1000.0).astype(np.float16))

        # user time:
        ptime_session = ptime.copy()
        ptime_session[~m_valid_data_session] = 0
        ptime_acc = ptime_session.sum(axis=1)

        session_mean = ptime_acc / session_count
        session_mean[no_session_data] = 0
        f.add(f"utime_t_{t_acc_max}", (session_mean.ravel()/1000.0).astype(np.float16))

        # idle time
        idle_time_session = diff_timestamp.astype(np.float32) - ptime.astype(np.float32)
        idle_time_session[~m_valid_data_session] = 0
        idle_time_acc = idle_time_session.sum(axis=1)

        session_mean = idle_time_acc / session_count
        session_mean[no_session_data] = 0
        f.add(f"idle_t_{t_acc_max}", (session_mean.ravel()/1000.0).astype(np.float16))


    # FEATURE: MEAN GLOBAL PROCESSING TIME FOR POSED QUESTIONS ##############################


    p_time[~m_VALID_DATA] = 0
    p_time_acc = p_time.cumsum(axis=1)

    # FEATURE: MEAN USER PROCESSING TIME FOR POSED QUESTIONS ##############################

    ptime[~m_VALID_DATA] = 0
    ptime_acc = ptime.cumsum(axis=1)

    # FEATURE: MEAN IDLE TIME ##########################################################

    idle_time[~m_VALID_DATA] = 0
    idle_time_acc = idle_time.cumsum(axis = 1)

    nCount = m_VALID_DATA.sum(axis=1)
    nCount[m_no_data] = 0

    for i in L_SAMPLING_POSITIONS:
        # Prepare cut counters with guard against division by zero.
        nCount_i = nCount.copy()
        nCount_i[nCount_i >i] = i

        nCount_i[m_no_data] = 1


        mean_i = p_time_acc[:, i-1] / nCount_i
        mean_i[m_no_data] = 0

        f.add(f"ptime_p_{i}",(mean_i.ravel()/1000.0).astype(np.float16))

        mean_i = ptime_acc[:, i - 1] / nCount_i
        mean_i[m_no_data] = 0
        f.add(f"utime_p_{i}", (mean_i.ravel()/1000.0).astype(np.float16))

        mean_i = idle_time_acc[:, i - 1] / nCount_i
        mean_i[m_no_data] = 0
        f.add(f"idle_p_{i}", (mean_i.ravel()/1000.0).astype(np.float16))
        # ptime_acc

    # FEATURE: CURRENT DIFF TIME
    diff_time = timestamp_current - prior_timestamp.astype(np.int64).ravel()
    diff_time[m_no_data] = 0
    f.add("diff_time", (diff_time.ravel()/1000.0).astype(np.float16))


    # FEATURE: CURRENT IDLE TIME
    current_idle_time = diff_time - ptime_current
    current_idle_time[m_no_data] = 0
    f.add("idle_time", (current_idle_time.ravel()/1000.0).astype(np.float16))


    # FEATURE: PREVIOUS NUMBER OF ATTEMPTS ON THE POSED QUESTION
    m = (np.repeat(content_id_current, u._width).reshape(-1, u._width) == data_questions)

    num_attempts = m.sum(axis=1)
    num_attempts[m_no_data] = 0
    f.add("num_attempts", num_attempts.ravel().astype(np.float16))

    # FEATURE: NUMBER OF CORRECT AND TOTAL NUMBER OF QUESTIONS POSED

    nCount = m_VALID_DATA.sum(axis=1)
    nCount[m_no_data] = 0

    m_correct = np.zeros(shape=m_VALID_DATA.shape, dtype=np.bool)

    m_correct[VALID_IDX] = (data_answers[VALID_IDX] == d_a[data_questions[VALID_IDX]])

    num_correct = m_correct.sum(axis=1)
    num_correct[m_no_data] = 0
    # f.add("num_correct", num_correct.ravel().astype(np.float16))

    nIncorrect = nCount - num_correct
    nIncorrect[m_no_data] = 0
    f.add("nIncorrect", nIncorrect.ravel().astype(np.float16))

    # COUNTS WITH CUTS

    for i in L_SAMPLING_POSITIONS:
        num_correct_i = m_correct[:, :i].sum(axis=1)
        num_correct_i[m_no_data] = 0
        f.add(f"num_correct_{i}", num_correct_i.ravel().astype(np.float16))

        nCount_i = nCount.copy()
        nCount_i[nCount_i > i] = i

        f_mean_user_accuracy, hmean_user_content_accuracy = get_accuracy_features(mean_question_accuracy, nCount_i, num_correct_i)

        f.add(f"mean_user_accuracy_{i}", f_mean_user_accuracy.ravel().astype(np.float16))
        f.add(f"hmean_user_content_accuracy_{i}", hmean_user_content_accuracy.ravel().astype(np.float16))


    # Sessions
    m = (diff_timestamp == 500000)
    valid_range = m.cumsum(axis=1) == 0
    m_valid_data_session = m_VALID_DATA & valid_range

    valid_idx_session = np.where(m_valid_data_session)

    session_specific(m_valid_data_session, valid_idx_session, "max_diff", f, data_questions, data_answers, mean_question_accuracy, m_no_data)

    # Session recent (sum 200000)
    m = diff_timestamp.cumsum(axis =1) < 200000
    m_valid_data_session = m_VALID_DATA & m
    valid_idx_session = np.where(m_valid_data_session)

    session_specific(m_valid_data_session, valid_idx_session, "acc_200", f, data_questions, data_answers, mean_question_accuracy, m_no_data)

    # Session recent (sum 500000)
    m = diff_timestamp.cumsum(axis=1) < 500000
    m_valid_data_session = m_VALID_DATA & m
    valid_idx_session = np.where(m_valid_data_session)

    session_specific(m_valid_data_session, valid_idx_session, "acc_500", f, data_questions, data_answers, mean_question_accuracy, m_no_data)

    # Session recent (sum 1000000)
    m = diff_timestamp.cumsum(axis=1) < 1000000
    m_valid_data_session = m_VALID_DATA & m
    valid_idx_session = np.where(m_valid_data_session)

    session_specific(m_valid_data_session, valid_idx_session, "acc_1000", f, data_questions, data_answers, mean_question_accuracy, m_no_data)

    # Session recent (sum 5000000)
    m = diff_timestamp.cumsum(axis=1) < 5000000
    m_valid_data_session = m_VALID_DATA & m
    valid_idx_session = np.where(m_valid_data_session)

    session_specific(m_valid_data_session, valid_idx_session, "acc_5000", f, data_questions, data_answers, mean_question_accuracy, m_no_data)

    return f.get_data_frame()

################################################################
#
#       store_previous
#

def store_previous(df_previous, u, u_t):

    anUser = df_previous.user_id.values
    iSort = np.argsort(anUser, kind = 'stable')

    anUser = anUser[iSort]

    content_id = df_previous.content_id.values[iSort].astype(np.uint32)
    user_answer = df_previous.user_answer.values[iSort].astype(np.uint32)
    timestamp = df_previous.timestamp.values[iSort]
    ptime = df_previous.prior_question_elapsed_time.values[iSort]

    anTest, idx_start, count = np.unique(anUser, return_counts= True, return_index= True)

    assert u._next_free == u_t._next_free

    u.ensure_users(anTest)
    u_t.ensure_users(anTest)

    assert u._next_free == u_t._next_free

    t_aiLoc = u_t.get_users_must_exist(anTest)

    t_aiOffset = t_aiLoc * u_t._width

    prior_timestamp = u_t._anData.ravel()[t_aiOffset]

    prior_timestamp[prior_timestamp == np.iinfo(u_t._datatype).max] = 0

    prior_timestamp = (np.repeat(prior_timestamp, count))

    diff_timestamp = timestamp - prior_timestamp

    ptime[np.isnan(ptime)] = 0.0

    ptime[ptime < 0] = 0.0
    ptime[ptime > 150000] = 150000.0

    ptime = ptime / 1000

    ptime = ptime.astype(np.uint32)

    diff_timestamp[diff_timestamp < 0] = 0

    diff_timestamp[diff_timestamp > 500000] = 500000

    diff_timestamp = diff_timestamp / 1000

    diff_timestamp = diff_timestamp.astype(np.uint32)

    assert np.log2(ptime_t)+ np.log2(diff_timestamp_t)+ np.log2(user_answer_t) + np.log2(content_id_t) < 32.0

    data = diff_timestamp + diff_timestamp_t * (ptime + ptime_t * (user_answer + user_answer_t * content_id.astype(np.uint32)))

    count_max = np.max(count)

    for iCount in range (0, count_max):

        m = (count >= (iCount + 1))
       
        idx_value = idx_start[m] + iCount

        anID_current = anUser[idx_value]
        anValue_current = data[idx_value]

        aiLoc = u.get_users_must_exist(anID_current)

        iOffset = aiLoc * u._width + u._anNext[aiLoc]

        u._anData.ravel()[iOffset] = anValue_current

        u._anNext[aiLoc] = (u._anNext[aiLoc] + 1) % u._width

    # Push current timestamp. u_t is single slot, offset is always zero.
    u_t._anData.ravel()[t_aiOffset] = timestamp[idx_start]



