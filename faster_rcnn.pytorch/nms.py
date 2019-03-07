import numpy as np
import sys

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]  #xmin
    y1 = dets[:, 1]  #ymin
    x2 = dets[:, 2]  #xmax
    y2 = dets[:, 3]  #ymax
    scores = dets[:, 4]  #confidence

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  #the size of bbox
    order = scores.argsort()[::-1]  #sort bounding boxes by decreasing order, returning array([3, 1, 2, 0])

    keep = []        # store the final bounding boxes
    while order.size > 0:
        i = order[0]   #the index of the bbox with highest confidence
        keep.append(i)    #save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]]) #array([ 257.,  280.,  255.])
        yy1 = np.maximum(y1[i], y1[order[1:]]) #array([ 118.,  135.,  118.])
        xx2 = np.minimum(x2[i], x2[order[1:]]) #array([ 360.,  360.,  358.])
        yy2 = np.minimum(y2[i], y2[order[1:]]) #array([ 235.,  235.,  235.])

        w = np.maximum(0.0, xx2 - xx1 + 1)   #array([ 104.,   81.,  104.])
        h = np.maximum(0.0, yy2 - yy1 + 1)   #array([ 118.,  101.,  118.])
        inter = w * h   #array([ 12272.,   8181.,  12272.])

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

assert(len(sys.argv) == 2)
ann_file = sys.argv[1]

with open(ann_file) as f:
    content = f.readlines()

prev_label = None
lbl = []
lbl_dict = {}
det = np.zeros((len(content), 5))
for i, line in enumerate(content):
    label, prob, xmin, ymin, xmax, ymax = line[:-1].split()
    if float(prob) < 0.4:
        continue
    if label not in lbl_dict:
        lbl_dict[label] = []
    lbl_dict[label].append([float(xmin), float(ymin), float(xmax), float(ymax), float(prob)])

f = open(ann_file, 'w')
f.truncate()    
for lbl in lbl_dict:
    det = np.array(lbl_dict[lbl])
    index = py_cpu_nms(det, 0.3)
    for i in range(len(index)): 
        f.write('{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(lbl, det[i][4], det[i][0], det[i][1], det[i][2], det[i][3]))
f.close()





