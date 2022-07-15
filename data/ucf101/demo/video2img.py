import os
import cv2
import tqdm


def split_video(folder_dir, save_dir, frames_per_video=3):
    os.makedirs(save_dir,exist_ok=True)
    bar = tqdm.tqdm(os.listdir(folder_dir))
    for filename in bar:
        video_name = filename[:-4]
        video_path = r"{}\{}".format(folder_dir,filename)
        image_save_dir = r"{}\{}".format(save_dir, video_name)
        os.makedirs(image_save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_num = cap.get(7)
        bar.set_postfix({"正在分割视频":video_name,"总帧数":frame_num})
        count = 0
        frames = 0
        step = int((frame_num - 1) / frames_per_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            if frames==frames_per_video:
                break
            count += 1
            if count%step==0:
                cv2.imwrite(r'{}\{}_frames.jpg'.format(image_save_dir,frames),frame)
                frames+=1
        cap.release()


split_video(r'F:\第十五届全国大学生计算机设计大赛\展示视频\clip_video',
            r'C:\Users\22704\Desktop\frams_split',)