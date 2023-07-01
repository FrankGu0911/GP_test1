import logging
import cv2,time,torch
import numpy as np
logging.basicConfig(level=logging.INFO)

seg_tag = {
    0: [0,0,0], # Unlabeled
    1: [70,70,70], # Building
    2: [100,40,40], # Fence
    3: [55,90,80], # Other
    4: [220,20,60], # Pedestrian
    5: [153,153,153], # Pole
    6: [157,234,50], # RoadLine
    7: [128,64,128], # Road
    8: [244,35,232], # Sidewalk
    9: [107,142,35], # Vegetation
    10: [0,0,142], # Car
    11: [102,102,156], # Wall
    12: [220,220,0], # TrafficSign
    13: [70,130,180], # Sky
    14: [81,0,81],  # Ground
    15: [150,100,100], # Bridge
    16: [230,150,140], # RailTrack
    17: [180,165,180], # GuardRail
    18: [250,170,30], # TrafficLight
    19: [110,190,160], # Static
    20: [170,120,50], # Dynamic
    21: [45,60,150], # Water
    22: [145,170,100], # Terrain
    23: [255,0,0], # RedLight
    24: [255,255,0], # YellowLight
    25: [0,255,0], # GreenLight
    
}

def cvt_rgb_seg(seg:np.ndarray):
    for i in seg_tag:
      seg = np.where(seg == [i,i,i], np.array(seg_tag[i]), seg)

    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    return seg
    

max = 0
for i in range(467):
    seg_bev = cv2.imread('train/test/test_data/weather-0/routes_town01_long_w0_06_23_09_24_57/topdown/%04d.png'%i)
    seg_bev = cvt_rgb_seg(seg_bev)
    # cur = seg_bev.max() - seg_bev.min()
    # max = cur if cur > max else max
    # print(max)
    cv2.imshow('seg_bev', seg_bev)
    if i == 0:
        cv2.waitKey(0)
    else:
        cv2.waitKey(33)