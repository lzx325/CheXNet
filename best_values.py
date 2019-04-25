import pickle
import sys
import os
for f_name in sys.argv[1:]:
    with open(os.path.join(f_name,"history.pkl"),'rb') as f:
        history=pickle.load(f)
    print("name: %s, best auc: %.5f, best epoch: %d"%(f_name,history["best_dev_eval_vals"]["auroc_avg"],history["best_dev_eval_vals_epoch"]))
