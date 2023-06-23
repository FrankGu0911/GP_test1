import logging
import cv2,time,torch
import numpy as np
logging.basicConfig(level=logging.INFO)

seg_tag = {
    0: [0,0,0],
    1: [70,70,70],
    2: [100,40,40],
    3: [55,90,80],
    4: [220,20,60],
    5: [153,153,153],
    6: [157,234,50],
    7: [128,64,128],
    8: [244,35,232],
    9: [107,142,35],
    10: [0,0,142],
    11: [102,102,156],
    12: [220,220,0],
    13: [70,130,180],
    14: [81,0,81],
    15: [150,100,100],
    16: [230,150,140],
    17: [180,165,180],
    18: [250,170,30],
    19: [110,190,160],
    20: [170,120,50],
    21: [45,60,150],
    22: [145,170,100],
    
}

def cvt_rgb_seg(seg:np.ndarray):
    for i in seg_tag:
      seg = np.where(seg == [i,i,i], np.array(seg_tag[i]), seg)
    # seg[seg > 22] = (0,0,0)
    # for i in range(23, 50):
    #     seg = np.where(seg == [i,i,i], np.array([0,0,0]), seg)
    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg
    

max = 0
for i in range(404):
    seg_bev = cv2.imread('train/test/test_data/weather-0/data/routes_town01_long_w0_06_23_00_31_21/topdown/%04d.png'%i)
    seg_bev = cvt_rgb_seg(seg_bev)
    # cur = seg_bev.max() - seg_bev.min()
    # max = cur if cur > max else max
    # print(cur)
    cv2.imshow('seg_bev', seg_bev)
    if i == 0:
        cv2.waitKey(0)
    else:
        cv2.waitKey(33)