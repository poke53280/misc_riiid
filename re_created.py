

N_SKILL = 13523
MAX_SEQ = 450
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 344
BATCH_SIZE = 128
DROPOUT = 0.1

0.2 - 0.5 sampling


sakt_779.pth

self._T0 = 0.2
self._T1 = 0.5

###############################################################################
###############################################################################

LGBM. On 800-data:


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