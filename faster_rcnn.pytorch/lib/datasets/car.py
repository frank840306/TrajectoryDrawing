import datasets
import datasets.car
import os
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import model.utils.cython_bbox
import pickle
import subprocess
from model.utils.config import cfg
from .car_eval import voc_eval
 
class car(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'pedestrian', 'bicycle')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
 
        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}
 
        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
 
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
 
    def image_id_at(self, i):
        return i;
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images', index + ext)
       
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path
 
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
 
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            print('cache:', cache_file)
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
 
        gt_roidb = [self._load_car_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
 
        return gt_roidb
 
    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')
 
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb
	# modified by Frank 
        if 'test' not in self._image_set:
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print(len(roidb))
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))
 
        return roidb
 
    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()
 
        box_list = []
        for i in range(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
 
        return self.create_roidb_from_box_list(box_list, gt_roidb)
 
    def selective_search_IJCV_roidb(self):
        """
        eturn the database of selective search regions of interest.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))
 
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb
 
        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)

        return roidb
 
    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)
 
        top_k = self.config['top_k']
        box_list = []
        for i in range(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))
 
        return self.create_roidb_from_box_list(box_list, gt_roidb)
 
    def _load_car_annotation(self, index):
        """
        Load image and bounding boxes info from txt files.
        """
        # add myself
        if 'demo' in self._image_set:
            return {
                'boxes': np.zeros((1, 4), dtype=np.uint16),
                'gt_classes': np.zeros((1), dtype=np.uint32),
                'gt_overlaps': scipy.sparse.csr_matrix(np.zeros((1, 3), dtype=np.float32)),
                'flipped': False
            }
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        # print 'Loading: {}'.format(filename)
        # Parse groundtruth file
        with open(filename) as f:
            data = f.readlines()
       
        num_objs = len(data)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for ix, aline in enumerate(data):
            tokens = aline.strip().split()  
            if len(tokens) != 5:
                continue
            cls = float(tokens[0])    # this file uses 0 as the background
            x1 = float(tokens[1])
            y1 = float(tokens[2])
            x2 = float(tokens[3])
            y2 = float(tokens[4])
            # print(cls, x1, y1, x2, y2)
            gt_classes[ix] = cls       
            boxes[ix, :] = [x1, y1, x2, y2]
            overlaps[ix, int(cls)] = 1.0
 
        overlaps = scipy.sparse.csr_matrix(overlaps)
 
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}
 
    def _write_car_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())
 
        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} car results file'.format(cls))
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id
 
    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']
 
        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)
 
    def _do_python_eval(self, comp_id, output_dir='output'):
        annopath = os.path.join(self._devkit_path, 'data', 'Annotations', '{:s}.txt')
        imagesetfile = os.path.join(self._devkit_path, 'data', 'ImageSets', self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.4)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}.'.format(np.mean(aps)))
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}\n'.format(np.mean(aps)))
        
    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_car_results_file(all_boxes)
        # self._do_matlab_eval(comp_id, output_dir)
        if 'demo' not in self._image_set:
            self._do_python_eval(comp_id, output_dir) 
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
 
if __name__ == '__main__':
    d = datasets.car('train', '')
    res = d.roidb
    from IPython import embed; embed()
