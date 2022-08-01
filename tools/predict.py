import argparse
import sys
sys.path.append('../')
import torch
from PIL.Image import open
import os
from torchvideo import transforms

from tools.getclasslist import getlist


def get_arg():
    parser = argparse.ArgumentParser(description='classification parameter configuration(predict)')
    parser.add_argument(
        '-pw',
        type=str,
        default=r'D:\PythonCode\Pytorch-VideoUnderstand-master\work_dir\exp-hmdb51_resnet50_test_2022_7_9_20_13\checkpoints\best_f1.pth',
        help='the weight applied to predict'
    )
    parser.add_argument(
        '-pv',
        type=str,
        default=r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\hmdb51\split16\brush_hair\April_09_brush_hair_u_nm_np1_ba_goo_0',
        help="prediction video frames' path"
    )
    parser.add_argument(
        '-clsp',
        type=str,
        default=r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\hmdb51\classes.txt',
        help="classes.txt's path"
    )
    parser.add_argument(
        '-rs',
        type=int,
        default=224,
        help='resized shape of input tensor'
    )
    return parser.parse_args()


args = get_arg()
# ----------------------------------------------------------------------------------------------------------------------
# 对预测图片进行预处理-resize+totensor
transformer = transforms.Compose([
            transforms.ResizeVideo((args.rs,args.rs)),
            transforms.PILVideoToTensor(),
])
pil_list = []
for filename in os.listdir(args.pv):
    path = args.pv + '/' + filename
    pil_list.append(open(path))
video = transformer(pil_list)
video_input = video.reshape((-1, *video.shape))
classes = getlist(args.clsp)
# ----------------------------------------------------------------------------------------------------------------------
# 加载好模型
if torch.cuda.is_available():
    print("Predict on cuda and there are/is {} gpus/gpu all.".format(torch.cuda.device_count()))
    print("Device name:{}\nCurrent device index:{}.".format(torch.cuda.get_device_name(), torch.cuda.current_device()))
else:
    print("Predict on cpu.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(args.pw)
print("Load weight from the path:{}.".format(args.pw))
model = model.to(device)
video_input = video_input.to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 前向传播进行预测
output = model(video_input)
score, prediction = torch.max(output, dim=1)
pred_class = classes[prediction.reshape((1,))]
score_value = score.detach().cpu().numpy().tolist()[0]
# ----------------------------------------------------------------------------------------------------------------------
print(pred_class, score_value)


