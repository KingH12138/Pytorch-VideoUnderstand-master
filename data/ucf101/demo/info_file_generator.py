import pandas as pd
import os
from tqdm import tqdm
import cv2


def get_videodataset_infofile(folder_dir,frame_dir,refer_path,class_path):
    """
    生成两种文件classes.txt and refer.csv
    :param folder_dir: 视频文件夹地点
    :return:None
    """
    data = {"filename":[],"path":[],"sr":[],"allframes":[],"duration/s":[],"label":[]}
    classes = os.listdir(folder_dir)
    txt_content = ""
    for class_name in classes:
        txt_content = txt_content + "{}\n".format(class_name)
    with open(class_path, 'w') as f:
        f.write(txt_content)
    for class_name in tqdm(os.listdir(folder_dir)):
        class_path = folder_dir + '/' + class_name
        frame_class_path = frame_dir + '/' + class_name
        for filename in os.listdir(class_path):
            video_name = filename[:-4]
            path = class_path + '/' + filename
            frame_path = frame_class_path + '/' +video_name
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                sr = cap.get(5)
                framenumber = cap.get(7)
                duration = int(framenumber/sr)  # 单位/s
                label = classes.index(class_name)
                data['filename'].append(video_name)
                data['path'].append(frame_path)
                data['label'].append(label)
                data['sr'].append(sr)
                data['allframes'].append(framenumber)
                data['duration/s'].append(duration)
            else:
                RuntimeError("Video can't be opened!")
    df = pd.DataFrame(data=data)
    df.to_csv(refer_path,encoding='utf-8')


get_videodataset_infofile(r"F:\PycharmProjects\Pytorch-VideoUnderstand-master\data\ucf101\videos",
                          r'F:\PycharmProjects\Pytorch-VideoUnderstand-master\data\ucf101\splitframes',
                          r"F:\PycharmProjects\Pytorch-VideoUnderstand-master\data\ucf101\video_refer.csv",
                          r"F:\PycharmProjects\Pytorch-VideoUnderstand-master\data\ucf101\classes.txt")