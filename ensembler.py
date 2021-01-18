

PIPELINE:
---------

sakt_single:

    read tidy
    split out num_train data and create group_train

    train on data[:num_train]   (In the cloud)
    ? Unsure validation

sakt_single:# LOAD AND PREDICT ON TAIL
    load trained model
    create iterate dataset [num_train:]
    predict on this dataset => y_sakt

    store y_sakt with other line data to

    df_pred.to_pickle(dir / "sakt_oof.pkl")


LGBM
----

    Data created for full dataset as df_iterated and loaded accumulators userbank and userbank_t

    expanding notebook:
        Train on first 0..4 of df_iterated to create lgb model.

    expanding notebook: PREDICT OOF AND MERGE WITH SAKT
        Merge iterated data with original lines found in df_pred. (y_sakt)
        Predict on these lines to create y_lgbm

        Now got <data>, <y_sakt>, <y_lgbm>, <y>

* Verify score quality as reported in the meta dataset.
* Create meta set, take features from main set.
* Find weak and strong points for each not model
* Tidy pipeline. Consider more models.
* Create new features, tidy up in lgbm.
* Vs: Understand sakt (include timings, include answers?)










