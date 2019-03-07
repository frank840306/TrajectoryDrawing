import os
import glob

ann_dir = '../Annotations'
img_dir = '../Images'

# train_fs = glob.glob(os.path.join(ann_dir, 'E*')) + glob.glob(os.path.join(ann_dir, 'c*')) + glob.glob(os.path.join(ann_dir, 'D*')) + glob.glob(os.path.join(ann_dir, 'Moon_morning_3*')) + glob.glob(os.path.join(ann_dir, 'Moon_morning_4*'))

# test_fs = glob.glob(os.path.join(ann_dir, 'Moon_morning_2*'))

demo_fs = sorted(glob.glob(os.path.join('../dataset/car_type_0/data/Images', 'type_0*.png')))

print(len(demo_fs))


fps = ['demo_type_0.txt']
fs = [demo_fs]
for fname, f in zip(fps, fs):

    fp = open(fname, 'w') 
    fp.truncate()
    for f_line in f:
        fp.write(f_line.split('/')[-1].split('.')[0]+ '\n')
    fp.close()


