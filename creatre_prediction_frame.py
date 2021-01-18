
from time import perf_counter


#################################################################
#
#       get_accuracy_features
#

def get_accuracy_features(mean_question_accuracy, nCount, nCorrect):
    m_data = nCount > 0
    nCount[~m_data] = 1
    f_mean_user_accuracy = (nCorrect / nCount).astype(np.float32)
    nCount[~m_data] = 0

    m_data = (f_mean_user_accuracy + mean_question_accuracy) > 0

    idx_data = np.where(m_data)[0]

    hmean_user_content_accuracy = f_mean_user_accuracy.copy()
    hmean_user_content_accuracy[:] = 0.0

    hmean_user_content_accuracy[idx_data] = 2 * (f_mean_user_accuracy[idx_data] * mean_question_accuracy[idx_data]) / (
                f_mean_user_accuracy[idx_data] + mean_question_accuracy[idx_data])

    return f_mean_user_accuracy.astype(np.float16), hmean_user_content_accuracy.astype(np.float16)


#################################################################
#
#       create_prediction_frame
#

def create_prediction_frame(df_current, u, u_t):
    t_begin = perf_counter()

    m_lecture = (df_current.content_type_id != 0)
    df_pred = pd.DataFrame(df_current[['row_id', 'user_id', 'timestamp', 'task_container_id', 'content_id', 'prior_question_elapsed_time']])

    df_pred = df_pred[~m_lecture]

    t_post_pred_create = perf_counter()

    data = np.empty((df_pred.shape[0], 5), dtype = np.float16)

    data[:, 0] = d_correct_mean[df_pred.content_id.values].astype(np.float16)
    data[:, 1] = d_part[df_pred.content_id.values].astype(np.float16)
    data[:, 2] = d_bundle[df_pred.content_id.values].astype(np.float16)
    data[:, 3] = d_tag1[df_pred.content_id.values].astype(np.float16)
    data[:, 4] = d_tag2[df_pred.content_id.values].astype(np.float16)

    df_stats = pd.DataFrame(data = data, index = df_pred.index, columns =['mean_question_accuracy', 'part', 'bundle_id','tag1', 'tag2'])

    df_pred = pd.concat([df_pred, df_stats], axis=1)

    t_pre_qa = perf_counter()

    df_res = get_qa_features(df_pred, u, u_t)

    df_res = df_res.set_index(df_pred.index)

    t_post_qa = perf_counter()

    df_pred = pd.concat([df_pred, df_res], axis = 1)

    del df_res

    t_end = perf_counter()

    dt_create = t_post_pred_create - t_begin
    dt_assign = t_pre_qa - t_post_pred_create
    dt_qa = t_post_qa - t_pre_qa

    dt_all = t_end - t_begin

    return df_pred, dt_create, dt_assign, dt_qa, dt_all

