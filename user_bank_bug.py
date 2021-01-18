


import pandas as pd
import random
import gc
from pathlib import Path

import pickle

random.seed(1)

DATA_DIR = "C:\\Users\\T149900\\Downloads\\riiid-test-answer-prediction\\"

dir = Path(DATA_DIR)
assert dir.is_dir()


u_kag = pickle.load(open(str(dir / "user_bank_train_kAG.pkl"), "rb"))


u_kag._anData.shape



u = pickle.load(open(str(dir / "user_bank_train.pkl"), "rb"))

u._anData.shape