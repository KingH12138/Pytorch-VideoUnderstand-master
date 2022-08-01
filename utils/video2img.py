import cv2
import os
import shutil
from tqdm import tqdm


def video2imgs(videoPath, imgPath, frames_per_video):
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)  # 目标文件夹不存在，则创建
    cap = cv2.VideoCapture(videoPath)  # 获取视频
    judge = cap.isOpened()  # 判断是否能打开成功
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率，视频每秒展示多少张图片
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    stride = int((frame_count + 1) / frames_per_video)
    frames = 0  # 用于统计所有帧数
    count = 0  # 用于统计保存的图片数量
    # h,w,c = cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FRAME_WIDTH),3
    while judge:
        if count == frames_per_video:
            break
        flag, frame = cap.read()  # 读取每一张图片 flag表示是否读取成功，frame是图片
        if not flag:
            break
        else:
            if frames % stride == 0:
                imgname = "%05d.jpg" % count
                newPath = imgPath + '/' + imgname
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                count += 1
        frames += 1
    cap.release()
    return count


def split(src_dir,save_dir,fn_per_video):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for class_name in os.listdir(src_dir):
        video_dir = src_dir + '/' + class_name
        bar = tqdm(os.listdir(video_dir))
        for filename in bar:
            videopath = video_dir + '/' + filename
            videoname = filename.split('.')[0]
            image_save_dir = save_dir + '/' + class_name + '/' + videoname
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            count = video2imgs(videopath, image_save_dir, fn_per_video)   ########
            bar.set_postfix({"Count": count})
    # check and delete
    print(save_dir)
    for cls_name in os.listdir(save_dir):
        cls_path = save_dir + '/' + cls_name
        for video_name in os.listdir(cls_path):
            video_frame_path = cls_path + '/' + video_name
            if len(os.listdir(video_frame_path))!=fn_per_video:
                print(video_frame_path)
                shutil.rmtree(video_frame_path)
