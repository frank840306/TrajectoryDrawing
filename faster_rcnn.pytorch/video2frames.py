import sys
import cv2
import os

assert(len(sys.argv) == 2)
demo_video = './demo_video'

vt = sys.argv[1]
if vt == '0':
    videos = ['Moon_morning_2_1.mp4', 'Moon_morning_2_3.mp4']
    # videos = ['Moon_morning_2_1.mp4', 'Moon_morning_2_2.mp4', 'Moon_morning_2_3.mp4']
else:
    videos = ['Fuxing_Xinhai_3_1.mp4', 'Fuxing_Xinhai_3_2.mp4']


for i, v in enumerate(videos):
    print(v)
    videocap = cv2.VideoCapture(os.path.join(demo_video, v))
    success, img = videocap.read()
    cnt = 0
    # fdir = v.split('.')[0]
    fdir = os.path.join('demo_video', v.split('.')[0])
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    while success:
        print('frame: {}'.format(cnt))
        cv2.imwrite(os.path.join(fdir, 'type_{}_video_{}_frame_{}.png'.format(vt, i + 1, cnt)), img)
        success, img = videocap.read()
        cnt += 1
