import os
import pandas as pd
from tqdm import tqdm


def getclslist(cls_path):
    with open(cls_path,'r') as f:
        content = f.read().split('\n')[:-1]
        return content


def generator(csv_path, frame_dir, cls_path):
    data = {"videoname": [], "framepath":[],"frame_num": [], "label": []}
    # 没有就写一个
    if os.path.exists(cls_path)==False:
        cls_list = os.listdir(frame_dir)
        cls_str = ""
        for cls in cls_list:
            cls_str = cls_str + "{}\n".format(cls)
        with open(cls_path,'w') as f:
            f.write(cls_str)
    cls_list = getclslist(cls_path)
    for cls_name in tqdm(os.listdir(frame_dir)):
        cls_path = frame_dir + '/' + cls_name
        for frame_name in os.listdir(cls_path):
            frame_path = cls_path + '/' + frame_name
            data['videoname'].append(frame_name)
            data['framepath'].append(frame_path)
            data['frame_num'].append(len(os.listdir(frame_path)))
            data['label'].append(cls_list.index(cls_name))
    df = pd.DataFrame(data=data)
    df.to_csv(csv_path,encoding='utf-8')


generator(r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\hmdb51\refer_16.csv',
          r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\hmdb51\split16',
          r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\hmdb51\classes.txt',)


