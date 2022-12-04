import numpy as np
import pandas as pd
import os


def summary_df(csv_path):
    df = pd.read_csv(os.path.join(csv_path, 'indicators.csv'), encoding='utf-8')
    name = df.columns.tolist()
    max_acc_id = df.idxmax()['Accuracy']
    for i in range(len(name)):
        print("%s:%.5f"%(name[i],df[name[i]][max_acc_id]))


summary_df(
    r'D:\PythonCode\Pytorch-VideoUnderstand-master\work_dir\class_action\temporal_attention\exp_8_12_20_41_class_action_resnet18(ta)_16'
)
