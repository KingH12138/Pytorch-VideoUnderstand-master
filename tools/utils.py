import cv2


def show_video(video_path,video_top5):
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        show_content = "{}:{}\n{}:{}\n{}:{}\n{}:{}\n{}:{}".format(
            'top2', video_top5['top2'],
            'top2', video_top5['top2'],
            'top3', video_top5['top3'],
            'top4', video_top5['top4'],
            'top5', video_top5['top5'],
        )
        cv2.putText(show_content)
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
