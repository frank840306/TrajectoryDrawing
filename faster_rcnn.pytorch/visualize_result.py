import matplotlib.pyplot as plt
import matplotlib.patches as pch
import numpy as np
import cv2
import os
import glob
import argparse

pedestrian_thres = 0.30
bicycle_thres = 0.30
MAX_LEN = 1000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', type=str, help='img directory')
    parser.add_argument('-a', '--ann_dir', type=str, help='ann directory')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory')
    parser.add_argument('-v', '--visualize', action='store_true', help='whether to visualize or not')
    return parser.parse_args()

def save_img(img_anns, img_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, img_file in enumerate(img_anns.keys()):      
        img = cv2.imread(os.path.join(img_dir, img_file))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        # fig, ax = plt.subplots(1)
        # ax.imshow(img)
        for ann_type in img_anns[img_file]:
            for ann in img_anns[img_file][ann_type]:
                if (ann_type[0] == 'p' and ann[4] < pedestrian_thres) or (ann_type[0] == 'b' and ann[4] < bicycle_thres):
                    continue
                # print('img: {}, type: {}, prob: {}'.format(img_file, ann_type, ann[4]))
                color = (255, 0, 0) if ann_type[0] == 'p' else (0, 0, 255)
                cv2.rectangle(img, (int(ann[0]), int(ann[1])), (int(ann[2]), int(ann[3])), color, 2)
                # rect = pch.Rectangle((ann[0], ann[1]), ann[2] - ann[0], ann[3] - ann[1], linewidth=1, edgecolor='r' if ann_type[0] == 'p' else 'b', facecolor='none')
                # ax.add_patch(rect)
                # plt.text(ann[0], ann[1], '[{}] Prob: {}'.format(ann_type, ann[4]))
        cv2.imwrite(os.path.join(output_dir, 'predict_' + img_file), img)
        # plt.savefig(os.path.join(output_dir, 'predict_' + img_file), bbox_inches='tight')
        if args.visualize:
            # plt.show()
            cv2.imshow('demo', img)
            cv2.waitKey()
        # plt.close()
        print('img {}: {}'.format(i+1, img_file))

def orgnize_ann(img_dir, ann_dir):
    img_anns = {}

    # for i, ann_file in enumerate(os.listdir(ann_dir)):
    for i, ann_file in enumerate(glob.glob(os.path.join(ann_dir, '*.txt'))):
        ann_type = ann_file.split('_')[-1].split('.')[0]
        print(ann_file)
        with open(ann_file) as f:
            anns = f.readlines()
        for ann in anns:
            img_file, prob, x_min, y_min, x_max, y_max = ann[:-1].split()
            img_file += '.png'
            prob = float(prob)
            x_min = float(x_min)
            y_min = float(y_min)
            x_max = float(x_max)
            y_max = float(y_max)
            if img_file not in img_anns:
                img_anns[img_file] = {}
            if ann_type not in img_anns[img_file]:
                img_anns[img_file][ann_type] = []    
            if len(img_anns[img_file][ann_type]) < MAX_LEN:
                img_anns[img_file][ann_type].append([x_min, y_min, x_max, y_max, prob])
        print('ann {}: {}'.format(i + 1, ann_file))
    return img_anns

def main(args):
    img_anns = orgnize_ann(args.img_dir, args.ann_dir)
    # img_files = map(lambda f: os.path.join(args.img_dir, f), img_anns.keys())
    # imgs = list(map(cv2.imread, img_files))
    # print(imgs)
    # img = cv2.imread(args.img_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # with open(args.ann_file) as f:
    #     ann = f.readlines()
    # fname = list(map(os.path.join(), ))
    
    
    save_img(img_anns, args.img_dir, args.output_dir)

if __name__ == "__main__":
    args = get_args()
    main(args)
    
    # example exe.
    # python visualize_reuslt.py -i ../dataset/car/data/Images/ -a ../dataset/car/results/test/ -o ../dataset/car/result_img -v
