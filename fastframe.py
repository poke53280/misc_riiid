

import numpy as np
import pandas as pd




f = FeatureBank(['a', 'b', 'c', 'd'], 15)

b_data = f.get_view('b')

b_data[3] = 2.9

f.get_data_frame()


def create_prediction_frame_OLD(df_current, u, u_t):

    t_begin = perf_counter()

    m_lecture = (df_current.content_type_id != 0)
    df_pred = pd.DataFrame(df_current[['row_id', 'user_id', 'timestamp', 'task_container_id', 'content_id', 'prior_question_elapsed_time']])

    df_pred = df_pred[~m_lecture]

    t_post_pred_create = perf_counter()

    part = d_part[df_pred.content_id.values].astype(np.float32)
    bundle = d_bundle[df_pred.content_id.values].astype(np.float32)
    tag1 = d_tag1[df_pred.content_id.values].astype(np.float32)
    tag2 = d_tag2[df_pred.content_id.values].astype(np.float32)
    mean_question_accuracy = d_correct_mean[df_pred.content_id.values].astype(np.float32)
    mean_question_ptime = d_ptime[df_pred.content_id.values].astype(np.float32)
    t_pre_qa = perf_counter()

    res = get_qa_features_OLD(df_pred.user_id.values, df_pred.content_id.values, df_pred.timestamp.values, df_pred.prior_question_elapsed_time.values, u, u_t)

    t_post_qa = perf_counter()

    Feature_rPMean = res[0]
    Feature_rUMean = res[1]
    Feature_rUMedian = res[2]
    Feature_Ldiff = res[3]
    Feature_Lidle = res[4]
    Feature_rUIDleMean = res[5]
    Feature_rUIDleMedian = res[6]
    Feature_nAttempts = res[7]
    nCorrect = res[8]
    nIncorrect = res[9]
    nCount = nCorrect + nIncorrect

    nCorrect3 = res[10]
    nCount3 = nCount.copy()
    nCount3[nCount3 > 3] = 3

    nCorrect30 = res[11]
    nCount30 = nCount.copy()
    nCount30[nCount30 > 30] = 30

    nCorrect100 = res[12]
    nCount100 = nCount.copy()
    nCount100[nCount100 > 100] = 100

    nCorrect150 = res[13]
    nCount150 = nCount.copy()
    nCount150[nCount150 > 150] = 150

    nCorrect200 = res[14]
    nCount200 = nCount.copy()
    nCount200[nCount200 > 200] = 200

    nCorrect250 = res[15]
    nCount250 = nCount.copy()
    nCount250[nCount250 > 250] = 250

    nCorrect300 = res[16]
    nCount300 = nCount.copy()
    nCount300[nCount300 > 300] = 300

    nCorrect350 = res[17]
    nCount350 = nCount.copy()
    nCount350[nCount350 > 350] = 350


    f_mean_user_accuracy, hmean_user_content_accuracy = get_accuracy_features(mean_question_accuracy, nCount, nCorrect)
    f_mean_user_accuracy_3, hmean_user_content_accuracy_3 = get_accuracy_features(mean_question_accuracy, nCount3, nCorrect3)
    f_mean_user_accuracy_30, hmean_user_content_accuracy_30 = get_accuracy_features(mean_question_accuracy, nCount30, nCorrect30)
    f_mean_user_accuracy_100, hmean_user_content_accuracy_100 = get_accuracy_features(mean_question_accuracy, nCount100, nCorrect100)

    f_mean_user_accuracy_150, hmean_user_content_accuracy_150 = get_accuracy_features(mean_question_accuracy, nCount150, nCorrect150)
    f_mean_user_accuracy_200, hmean_user_content_accuracy_200 = get_accuracy_features(mean_question_accuracy, nCount200, nCorrect200)
    f_mean_user_accuracy_250, hmean_user_content_accuracy_250 = get_accuracy_features(mean_question_accuracy, nCount250, nCorrect250)
    f_mean_user_accuracy_300, hmean_user_content_accuracy_300 = get_accuracy_features(mean_question_accuracy, nCount300, nCorrect300)
    f_mean_user_accuracy_350, hmean_user_content_accuracy_350 = get_accuracy_features(mean_question_accuracy, nCount350, nCorrect350)


    t_preassign = perf_counter()


    df_pred = df_pred.assign(mean_question_accuracy = mean_question_accuracy)
    df_pred = df_pred.assign(num_attempts = Feature_nAttempts)
    df_pred = df_pred.assign(num_correct = nCorrect)

    df_pred = df_pred.assign(num_total = nCount)
    df_pred = df_pred.assign(mean_user_accuracy = f_mean_user_accuracy)
    df_pred = df_pred.assign(hmean_user_content_accuracy = hmean_user_content_accuracy)


    # 3
    df_pred = df_pred.assign(num_correct_3=nCorrect3)
    df_pred = df_pred.assign(mean_user_accuracy_3=f_mean_user_accuracy_3)
    df_pred = df_pred.assign(hmean_user_content_accuracy_3=hmean_user_content_accuracy_3)

    # 30
    df_pred = df_pred.assign(num_correct_30=nCorrect30)
    df_pred = df_pred.assign(mean_user_accuracy_30=f_mean_user_accuracy_30)
    df_pred = df_pred.assign(hmean_user_content_accuracy_30=hmean_user_content_accuracy_30)

    # 100
    df_pred = df_pred.assign(num_correct_100=nCorrect100)
    df_pred = df_pred.assign(mean_user_accuracy_100=f_mean_user_accuracy_100)
    df_pred = df_pred.assign(hmean_user_content_accuracy_100=hmean_user_content_accuracy_100)

    # 150
    df_pred = df_pred.assign(num_correct_150=nCorrect150)
    df_pred = df_pred.assign(mean_user_accuracy_150=f_mean_user_accuracy_150)
    df_pred = df_pred.assign(hmean_user_content_accuracy_150=hmean_user_content_accuracy_150)

    # 200
    df_pred = df_pred.assign(num_correct_200=nCorrect200)
    df_pred = df_pred.assign(mean_user_accuracy_200=f_mean_user_accuracy_200)
    df_pred = df_pred.assign(hmean_user_content_accuracy_200=hmean_user_content_accuracy_200)

    # 250
    df_pred = df_pred.assign(num_correct_250=nCorrect250)
    df_pred = df_pred.assign(mean_user_accuracy_250=f_mean_user_accuracy_250)
    df_pred = df_pred.assign(hmean_user_content_accuracy_250=hmean_user_content_accuracy_250)

    # 300
    df_pred = df_pred.assign(num_correct_300=nCorrect300)
    df_pred = df_pred.assign(mean_user_accuracy_300=f_mean_user_accuracy_300)
    df_pred = df_pred.assign(hmean_user_content_accuracy_300=hmean_user_content_accuracy_300)

    # 350
    df_pred = df_pred.assign(num_correct_350=nCorrect350)
    df_pred = df_pred.assign(mean_user_accuracy_350=f_mean_user_accuracy_350)
    df_pred = df_pred.assign(hmean_user_content_accuracy_350=hmean_user_content_accuracy_350)

    df_pred = df_pred.assign(pTime_mean=Feature_rPMean)
    df_pred = df_pred.assign(uTime_mean=Feature_rUMean)
    df_pred = df_pred.assign(uTime_median=Feature_rUMedian)

    df_pred = df_pred.assign(diff_time=Feature_Ldiff)
    df_pred = df_pred.assign(idle_time=Feature_Lidle)

    df_pred = df_pred.assign(uidle_mean=Feature_rUIDleMean)
    df_pred = df_pred.assign(uidle_median=Feature_rUIDleMedian)

    df_pred = df_pred.assign(part = part)
    df_pred = df_pred.assign(bundle_id=bundle)

    df_pred = df_pred.assign(tag1=tag1)
    df_pred = df_pred.assign(tag2=tag2)


    t_end = perf_counter()


    dt_create = t_post_pred_create - t_begin
    dt_assign = t_pre_qa - t_post_pred_create
    dt_qa = t_post_qa - t_pre_qa

    dt_all = t_end - t_begin

    return df_pred, dt_create, dt_assign, dt_qa, dt_all




def get_qa_features_OLD(user_id_current, content_id_current, timestamp_current, ptime_current, u, u_t):

    QUESTION_MAX = len(d_a)

    iLoc = u.get_users(user_id_current)

    m_no_data = iLoc < 0

    if m_no_data.all():
        fZeros = np.zeros(user_id_current.shape[0], dtype = np.float32)
        return fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros, fZeros

    # Set to in range. Data is invalidated later.
    iLoc[m_no_data] = 0

    d = u._anData[iLoc]

    m_VALID_DATA = (d < np.iinfo(u._datatype).max)
    VALID_IDX = np.where(m_VALID_DATA)

    # Valid mask: Last N elements

    m_last_3 = get_last_n_mask(u, 3, iLoc) & m_VALID_DATA
    m_last_30 = get_last_n_mask(u, 30, iLoc) & m_VALID_DATA
    m_last_100 = get_last_n_mask(u, 100, iLoc) & m_VALID_DATA

    m_last_150 = get_last_n_mask(u, 150, iLoc) & m_VALID_DATA
    m_last_200 = get_last_n_mask(u, 200, iLoc) & m_VALID_DATA
    m_last_250 = get_last_n_mask(u, 250, iLoc) & m_VALID_DATA
    m_last_300 = get_last_n_mask(u, 300, iLoc) & m_VALID_DATA
    m_last_350 = get_last_n_mask(u, 350, iLoc) & m_VALID_DATA


    prior_timestamp = u_t._anData[iLoc]

    diff_timestamp = (d % diff_timestamp_t) * 1000

    d = d // diff_timestamp_t

    ptime = (d % ptime_t) * 1000

    d = d // ptime_t

    data_answers = d % user_answer_t
    data_questions = d // user_answer_t

    # FEATURE: MEAN GLOBAL PROCESSING TIME FOR POSED QUESTIONS ##############################

    p_time = data_questions.astype(np.float32).copy()

    p_time[VALID_IDX] = d_ptime[data_questions[VALID_IDX]]

    masked_p_time = p_time.view(np.ma.MaskedArray)
    masked_p_time.mask = ~m_VALID_DATA

    Feature_rPMean = masked_p_time.mean(axis = 1).reshape(p_time.shape[0], -1).data
    Feature_rPMean[m_no_data] = 0

    # FEATURE: MEAN AND MEDIAN USER PROCESSING TIME FOR POSED QUESTIONS ##############################

    masked_utime = ptime.view(np.ma.MaskedArray)
    masked_utime.mask = ~m_VALID_DATA

    Feature_rUMean = masked_utime.mean(axis = 1).reshape(masked_utime.shape[0], -1).data
    Feature_rUMean[m_no_data] = 0.5

    Feature_rUMedian = np.ma.median(masked_utime, axis=1).reshape(masked_utime.shape[0], -1).data
    Feature_rUMedian[m_no_data] = 0

    # FEATURE: CURRENT DIFF TIME
    Feature_Ldiff = timestamp_current - prior_timestamp.astype(np.int64).ravel()
    Feature_Ldiff[m_no_data] = 0

    # FEATURE: CURRENT IDLE TIME
    Feature_Lidle = Feature_Ldiff - ptime_current
    Feature_Lidle[m_no_data] = 0

    # FEATURE: MEAN AND MEDIAN IDLE TIME
    idle_time = diff_timestamp.astype(np.float32) - ptime.astype(np.float32)

    masked_idle = idle_time.view(np.ma.MaskedArray)
    masked_idle.mask = ~m_VALID_DATA

    Feature_rUIDleMean = masked_idle.mean(axis= 1).reshape(masked_idle.shape[0], -1).data
    Feature_rUIDleMean[m_no_data] = 0

    Feature_rUIDleMedian = np.ma.median(masked_idle, axis=1).reshape(masked_idle.shape[0], -1).data
    Feature_rUIDleMedian[m_no_data] = 0

    # FEATURE: PREVIOUS NUMBER OF ATTEMPTS ON THE POSED QUESTION
    m = (np.repeat(content_id_current, u._width).reshape(-1, u._width) == data_questions)

    Feature_nAttempts = m.sum(axis= 1)
    Feature_nAttempts[m_no_data] = 0

    # FEATURE: NUMBER OF CORRECT AND TOTAL NUMBER OF QUESTIONS POSED

    nCount = m_VALID_DATA.sum(axis = 1)

    m_correct = np.zeros(shape = m_VALID_DATA.shape, dtype = np.bool)

    m_correct[VALID_IDX] = (data_answers[VALID_IDX] == d_a[data_questions[VALID_IDX]])

    nCorrect = m_correct.sum(axis = 1)
    nCorrect[m_no_data] = 0

    nIncorrect = nCount - nCorrect


    masked_correct = m_correct.view(np.ma.MaskedArray)

    masked_correct.mask = ~(m_VALID_DATA & m_last_3)
    nCorrect_3 = masked_correct.sum(axis = 1).data
    nCorrect_3[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_30)
    nCorrect_30 = masked_correct.sum(axis=1).data
    nCorrect_30[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_100)
    nCorrect_100 = masked_correct.sum(axis=1).data
    nCorrect_100[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_150)
    nCorrect_150 = masked_correct.sum(axis=1).data
    nCorrect_150[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_200)
    nCorrect_200 = masked_correct.sum(axis=1).data
    nCorrect_200[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_250)
    nCorrect_250 = masked_correct.sum(axis=1).data
    nCorrect_250[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_300)
    nCorrect_300 = masked_correct.sum(axis=1).data
    nCorrect_300[m_no_data] = 0

    masked_correct.mask = ~(m_VALID_DATA & m_last_350)
    nCorrect_350 = masked_correct.sum(axis=1).data
    nCorrect_350[m_no_data] = 0

    return Feature_rPMean.astype(np.float32), Feature_rUMean.astype(np.float32), Feature_rUMedian.astype(np.float32), Feature_Ldiff.astype(np.float32), Feature_Lidle.astype(np.float32), Feature_rUIDleMean.astype(np.float32), Feature_rUIDleMedian.astype(np.float32), Feature_nAttempts.astype(np.float32), nCorrect.astype(np.float32), nIncorrect.astype(np.float32), nCorrect_3.astype(np.float32), nCorrect_30.astype(np.float32), nCorrect_100.astype(np.float32), nCorrect_150.astype(np.float32), nCorrect_200.astype(np.float32), nCorrect_250.astype(np.float32), nCorrect_300.astype(np.float32), nCorrect_350.astype(np.float32)


