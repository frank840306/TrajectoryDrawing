import argparse
import glob
import os
import cv2
import copy
import numpy as np
from collections import deque
THRES_IOU = 0.0

def get_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--type', default=0, type=int)
	parser.add_argument('--video', default=3, type=int)
	parser.add_argument('--start', default=890, type=int)
	parser.add_argument('--end', default=1080, type=int)
	parser.add_argument('--object', default='pedestrian', type=str)
	parser.add_argument('--object_idx', default=0, type=int)
	parser.add_argument('--output_dir', default='/tmp2/frank840306/demo_video', type=str)

	return parser.parse_args()

def mid_point(bbox):
	x = (bbox[0] + bbox[2]) / 2
	y = (bbox[1] + bbox[3]) / 2
	return (x, y)

def iou(bbox1, bbox2):
	one_x = bbox1[0]
	one_y = bbox1[1]
	one_w = bbox1[2] - bbox1[0]
	one_h = bbox1[3] - bbox1[1]
	
	two_x = bbox2[0]
	two_y = bbox2[1]
	two_w = bbox2[2] - bbox2[0]
	two_h = bbox2[3] - bbox2[1]

	if((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
		lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
		lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))
 
		rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
		rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))
 
		inter_w = abs(rd_x_inter - lu_x_inter)
		inter_h = abs(lu_y_inter - rd_y_inter)
 
		inter_square = inter_w * inter_h
		union_square = (one_w * one_h) + (two_w * two_h) - inter_square
 
		IOU = inter_square / union_square * 1.0
		#print("calcIOU:", IOU)
	else:
		IOU = 0
		#print("No intersection!")
 
	return IOU


def read_bbox(bbox_dir, cfg):
	bbox_type = ['pedestrian', 'bicycle']
	object_dict = {}
	for bt in bbox_type:
		if bt not in object_dict:
			object_dict[bt] = {}
		bbox_file = os.path.join(bbox_dir, 'comp4_det_demo_type_{}_{}.txt'.format(cfg['type'], bt))
		with open(bbox_file) as f:
			content = f.readlines()
		for line in content:
			fname, prob, xmin, ymin, xmax, ymax = line[:-1].split()
			frame = int(fname.split('_')[-1])
			if frame > cfg['fend'] or frame < cfg['fstart']:
				continue
			#if frame < 1080 and frame > 870:
			#	print(fname)
			if int(fname[13]) != cfg['video']:
				continue

			if fname not in object_dict[bt]:
				object_dict[bt][fname] = []
			#print(xmin, ymin, xmax, ymax)
			object_dict[bt][fname].append([float(xmin), float(ymin), float(xmax), float(ymax), float(prob)])
			#print(object_dict[bt][fname][-1])
	return object_dict

def trace(cfg):
	bbox_dir = os.path.join('../dataset', 'car_type_{}'.format(cfg['type']), 'results', 'demo_type_{}'.format(cfg['type']))

	points = []
	bbox = read_bbox(bbox_dir, cfg)
        
	#print(bbox['bicycle']['type_0_video_3_frame_870'])
	#print(bbox['bicycle']['type_0_video_3_frame_871'])
	#print(bbox['bicycle']['type_0_video_3_frame_872'])
	#print(bbox['bicycle']['type_0_video_3_frame_873'])
	#print(bbox['bicycle']['type_0_video_3_frame_874'])
	prev_bbox = bbox[cfg['object']]['type_{}_video_{}_frame_{}'.format(cfg['type'], cfg['video'], cfg['fstart'])][cfg['object_idx']] # 26
	print('start: ', prev_bbox)
	points.append(mid_point(prev_bbox))
	for fidx in range(cfg['fstart'] + 1, cfg['fend']):
		frame = 'type_{}_video_{}_frame_{}'.format(cfg['type'], cfg['video'], fidx)
		if frame in bbox[cfg['object']]:
			
			bboxs = bbox[cfg['object']][frame]
			# print(bboxs)
			best_score = 0
			best_bbox = None
			for bx in bboxs:
				score = iou(prev_bbox, bx)
				if score > best_score:
					best_bbox = copy.deepcopy(bx)
					best_score = score
			if best_bbox is None or best_score < THRES_IOU:
				# no suitable bbox, use the prev one
				best_bbox = prev_bbox

			prev_bbox = best_bbox
		print(mid_point(prev_bbox))
		points.append(mid_point(prev_bbox))
	return points

def output_video(cfg, points):
	img_dir = os.path.join('../dataset', 'car_type_{}'.format(cfg['type']), 'data', 'Images')
	# glob.glob(os.path.join(img_dir, 'type_{}_video_{}_frame_{}'.format(cfg['type'], cfg['video'])))
	pts = deque(maxlen=200)
	

	img = cv2.imread(os.path.join(img_dir, 'type_{}_video_{}_frame_0.png'.format(cfg['type'], cfg['video'])))
	h, w, c = img.shape
	print(img.shape)
	out = cv2.VideoWriter(os.path.join(cfg['output_dir'], 'demo_type_{}_video_{}_movement.mp4'.format(cfg['type'], cfg['video'])), 0x7634706d, 30, (w, h))
	for idx in range(len(points)):
		fidx = cfg['fstart'] + idx
		img_file = os.path.join(img_dir, 'type_{}_video_{}_frame_{}.png'.format(cfg['type'], cfg['video'], fidx))
		img = cv2.imread(img_file)

		pts.appendleft((int(points[idx][0]), int(points[idx][1])))
		for i in range(1, len(pts)):
			if pts[i - 1] is None or pts[i] is None:
				continue
			thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
			#print(thickness)
			cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), thickness)
		out.write(img)
		print('write frame {} / {}'.format(idx, len(points)))
	out.release()

def main():
	args = get_args()
	cfg = {
		'type': args.type,
		'video': args.video,
		'fstart': args.start,
		'fend': args.end,
                'object': args.object,
                'object_idx': args.object_idx,
		'output_dir': args.output_dir
	}
	points = trace(cfg)
	
	# print('size: {}'.format(len(points)))
	# for i, p in enumerate(points):
	# 	print('frame: {}, ({}, {})'.format(i + 1, p[0], p[1]))
	output_video(cfg, points)
if __name__ == '__main__':
	main()
