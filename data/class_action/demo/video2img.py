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
    # # 静默处理
    # while count<frames_per_video:
    #     imgname = "%05d.jpg" % count
    #     newPath = imgPath + '/' + imgname
    #     cv2.imwrite(newPath,np.zeros(shape=(h,w,c)),[cv2.IMWRITE_JPEG_QUALITY, 100])
    cap.release()
    return count


def main():
    video_all_dir = r'D:\数据集\hmdb51_org'  ########
    save_dir = r'D:\PythonCodes\Pytorch-VideoUnderstand-master\data\hmdb51\splitframe_16'  ########

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for class_name in os.listdir(video_all_dir):
        video_dir = video_all_dir + '/' + class_name
        bar = tqdm(os.listdir(video_dir))
        for filename in bar:
            videopath = video_dir + '/' + filename
            videoname = filename.split('.')[0]
            image_save_dir = save_dir + '/' + class_name + '/' + videoname
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            count = video2imgs(videopath, image_save_dir, 16)   ########
            bar.set_postfix({"Count": count})


main()


def check(frames_dir, frames_per_video):
    for cls_name in os.listdir(frames_dir):
        cls_path = frames_dir + '/' + cls_name
        for video_name in os.listdir(cls_path):
            video_frame_path = cls_path + '/' + video_name
            if len(os.listdir(video_frame_path))!=frames_per_video:
                print(video_frame_path)
                shutil.rmtree(video_frame_path)


check(r'D:\PythonCodes\Pytorch-VideoUnderstand-master\data\hmdb51\splitframe_16', 16) ########
