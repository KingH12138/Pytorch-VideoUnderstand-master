# Pytorch-VideoUnderstand-master使用方法

文件构架
```
├─.idea
│  └─inspectionProfiles
├─data
│  ├─class_action
│  ├─hmdb51
│  └─ucf101
│      └─demo
│          └─__pycache__
├─datasets
│  └─__pycache__
├─images
├─models
│  └─__pycache__
├─tools
│  └─__pycache__
└─work_dir
```

## 一.配置环境


先创建好环境并安装好pytorch，激活环境(至于要安装什么pytorch版本，可以参考[这篇文章](https://blog.csdn.net/Killer_kali/article/details/123173414?spm=1001.2014.3001.5501)：

```

conda create -n torchvu python=3.7
conda activate torchvu
conda install pytorch=1.8 torchvision cudatoolkit=10.2

```

进入项目目录：

```
cd Pyotrch-AudioClassification-master
```

安装相关包库：

```
pip install -r requirements.txt
```

tips:

如果prettytable库无法安装，可以尝试如下命令：

```
python -m pip install -U prettytable
```

## 二.运行predict测试文件
将权重存放到任意路径下，打开pycharm项目tools下的predict.py文件。

修改get_arg下的参数——预测视频(已分帧)、权重文件以及类别信息文件的路径。运行即可
## 三.制作并训练自己的数据(目前仅支持folder存放格式)

folder:
```
-datasets:
    -class0
        -0.wav
        -1.wav
        .......
    -class1
        -0.wav
        -1.wav
        .......
    .......
```


1.在data下根据自己数据集的格式选择对应格式的demo;

2.运行分帧脚本并检查，之后运行对应的数据集信息生成脚本从而得到DIF等文件;

3.在train.py脚本文集里面修改get_arg函数下的参数;

4.运行train.py脚本即可;
