"""

This class is created to make folder dataset

easy to be processed by my demo.

image_dir format:

-JPEGImage
    -classname0
        -index0.jpg
        -index1.jpg
        ......
    -classname1
    ......
csv format:

index   filename filepath label
0       ...     ...     ...
1       ...     ...     ...
2       ...     ...     ...
3       ...     ...     ...
......

"""

import os
import pandas as pd
from tqdm import tqdm


class FolderVideoData():
    def __init__(self, sf_dir, cls_path, refer_path):
        self.sf_dir = sf_dir
        if len(os.listdir(self.sf_dir)) == 0:
            RuntimeError(f"There is no class directory in f{self.sf_dir},please check your directory "
                         f"and try the 'video2img' demo in 'utils")
        self.cls_path = cls_path
        self.refer_path = refer_path
        self.cls_list = os.listdir(self.sf_dir)

    def cls2txt(self):
        with open(self.cls_path, 'w') as f:
            content = '\n'.join(self.cls_list)
            f.write(content)

    def generate(self):
        csv_path = self.refer_path
        frame_dir = self.sf_dir
        data = {"name": [], "framepath": [], "label": []}
        # classes.txt
        self.cls2txt()
        # DIF
        for cls_name in tqdm(os.listdir(frame_dir)):
            cls_path = os.path.join(frame_dir, cls_name)
            for frame_name in os.listdir(cls_path):
                frame_path = os.path.join(cls_path, frame_name)
                data['name'].append(frame_name)
                data['framepath'].append(frame_path)
                data['label'].append(self.cls_list.index(cls_name))
        df = pd.DataFrame(data=data)
        df.to_csv(csv_path, encoding='utf-8')
