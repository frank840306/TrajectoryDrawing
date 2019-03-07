import sys
import glob
import os
import cv2

def get_args():
    if len(sys.argv) != 3:
        frame_dir = ''
        video_dir = '/tmp2/frank840306/demo_video/' 
    else:
        frame_dir = sys.argv[1]
        video_dir = sys.argv[2]
    print('frame_dir: {}, video_dir: {}'.format(frame_dir, video_dir))
    return frame_dir, video_dir

def frames2video(frame_dir, video_dir, fps):
    group_len = []
    if 'type_1' in frame_dir:
        # two video
        group_len.append(len(glob.glob(os.path.join(frame_dir, '*video_1*'))))
        group_len.append(len(glob.glob(os.path.join(frame_dir, '*video_2*'))))
        vt = 1
    elif 'type_0' in frame_dir:
        # three video
        group_len.append(len(glob.glob(os.path.join(frame_dir, '*video_1*'))))
        group_len.append(len(glob.glob(os.path.join(frame_dir, '*video_2*'))))
        group_len.append(len(glob.glob(os.path.join(frame_dir, '*video_3*'))))
        vt = 0
    for i, gl in enumerate(group_len):
        print('start to output video {}, frame_num: {}'.format(i + 1, gl))
        img = cv2.imread(os.path.join(frame_dir, 'predict_type_{}_video_{}_frame_0.png'.format(vt, i + 1)))
        h, w, c = img.shape
        # out = cv2.VideoWriter(os.path.join(video_dir, 'demo_type_{}_video_{}.mp4'.format(vt, i + 1)), cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
        out = cv2.VideoWriter(os.path.join(video_dir, 'demo_type_{}_video_{}.mp4'.format(vt, i + 1)), 0x7634706d, fps, (w, h))
        for idx in range(gl):
            fname = os.path.join(frame_dir, 'predict_type_{}_video_{}_frame_{}.png'.format(vt, i + 1, idx))
            img = cv2.imread(fname)
            out.write(img)
        out.release()


def main(frame_dir, video_dir):   
    frames2video(frame_dir, video_dir, 30)

if __name__ == '__main__':
    frame_dir, video_dir = get_args()
    main(frame_dir, video_dir)

