# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:00:28 2019

@author: mohamed achraf
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from Functions import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import pickle



with open('train.pickle', 'rb') as f:
    color_space, hog_channel, svc,orient, pix_per_cell, cell_per_block = pickle.load(f)

def detect_boxes(img):

    boxes = []
    
    ystart = 400
    ystop = 464
    scale = 1.0
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 416
    ystop = 480
    scale = 1.0
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 496
    scale = 1.5
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 528
    scale = 1.5
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 528
    scale = 2.0
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 560
    scale = 2.0
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 596
    scale = 3.5
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
    ystart = 464
    ystop = 660
    scale = 3.5
    boxes.append(find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, None, 
                           orient, pix_per_cell, cell_per_block, None, None))
 
    r = [item for sublist in boxes for item in sublist] 
    
    # add detections to the history
    if len(r) > 0:
        det.add_rects(r)
    
    heatmap_img = np.zeros_like(img[:,:,0])
    for rect_set in det.prev_rects:
        heatmap_img = add_heat(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(det.prev_rects)//2)
     
    labels = label(heatmap_img)
    draw_img= draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


det = Vehicle_Detect()
out_file = 'out3.mp4'
clip = VideoFileClip('project_video.mp4')
clip_out= clip.fl_image(detect_boxes)
clip_out.write_videofile(out_file, audio=False)
"""
cap = cv2.VideoCapture('project_video.mp4')
while True:
    #capture frame by frame
    ret, frame = cap.read()
    det = Vehicle_Detect()
    out = detect_boxes(frame)

    #display the resulting frame
    cv2.imshow('out', out)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
"""