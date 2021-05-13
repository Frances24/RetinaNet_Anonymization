# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:38:44 2021

@author: fran-
"""


import argparse
import os
import cv2
import time
import face_detection
import numpy as np
from PIL import Image
from blur import anonymize_face


def load_source(input_type, input_file):

    
    if input_type == 'video_file':
            if not os.path.isfile(input_file):
                print('Check file path')
            print('Loading video file')
            source = cv2.VideoCapture(input_file)
    else:
        if not os.path.isdir(input_file):
            print('Check file path')
        print('Loading images')
        src_list = sorted(list(map(lambda im: os.path.join(input_file, im), os.listdir(input_file))))
        source = iter(src_list)
            
    return source


def get_frame(input_type, input_file, source, i):
    if input_type == 'video_file':
        grabbed, img = source.read()
        fname = 'anon_{}.jpg'.format(i)
    else:
        try:
            grabbed = True
            fname =  next(source)
            img = cv2.imread(fname)
        
        except StopIteration:
            grabbed = False
            img = None
            fname = None
            
    return grabbed, img, fname
            
        
def resize_frame(frame, size=None):

    height, width = frame.shape[:2]

    if hasattr(size, '__iter__'):
        # size is a tuple 
        height, width = map(int, size)
    
    else:
        # resize long edge to size and preserve aspect ration
        ratio = size / height if height > width else size / width
        height = int(height * ratio)
        width = int(width * ratio)

    return cv2.resize(frame, (width, height))

def cap_bboxes(boxes, img):
    
    boxes[:, (0,2)] = np.minimum(boxes[:, (0,2)], np.array([[img.shape[1]]*len(boxes), [img.shape[1]]*len(boxes)]).transpose())
    boxes[:, (1,3)] = np.minimum(boxes[:, (1,3)], np.array([[img.shape[0]]*len(boxes), [img.shape[0]]*len(boxes)]).transpose())
    
    return boxes

def filter_boxes_by_size(boxes, scale, frame_size):

    w, h = frame_size
    
    keep_idx = np.where((boxes[:,2] - boxes[:,0]) <= w*scale)[0]
    boxes = boxes[keep_idx]
    
    keep_idx = np.where((boxes[:,3] - boxes[:,1]) < h*scale)[0]
    
    boxes = boxes[keep_idx]
    return boxes

def detect_faces(img, detector):
    
    dets = detector.detect(img[:,:, ::-1])
    scores = dets[:, 4]
    boxes = cap_bboxes(dets[:, :4], img)
    
    return boxes, scores


def anonymize_faces(boxes,img, anon_style):
    for bbox in boxes:
        bbox = [max(0,int(_)) for _ in bbox]
        if bbox[2]==0 or bbox[3]==0 or (bbox[2]-bbox[0])<=0 or (bbox[3]-bbox[1])<=0:
            continue
        anonymize_face(Image.fromarray(img), img, bbox, anon_style)
    
    return img


def load_writer(output_type, output_loc, input_file, img_size):
        if output_type == 'video':
            fourcc= cv2.VideoWriter_fourcc(*'MP4V')
            video_name = input_file.split('/')[-1].split('.')[0] + '_anon.mp4'
            fps = 10
            writer = cv2.VideoWriter(os.path.join(output_loc,video_name), fourcc, fps, img_size)
        else:
            writer = cv2.imwrite
            
        return writer


def write_to_file(img, output_file_type, output_file_loc, writer, fname):
        if not os.path.isdir(output_file_loc):
            os.mkdir(output_file_loc)
        if output_file_type == 'video':
            writer.write(img)
        else:
            # writer(os.path.join(output_file_loc, obj.input.split('\\')[-1].split('.')[0]+ '_anon{}.jpg'.format(idx)),img)
            writer(os.path.join(output_file_loc, fname.split('\\')[-1].split('.')[0]+ '.jpg'), img)
            
        return
    
def anonymize_input(opt):
    
    detector = face_detection.build_detector(
        opt.detector,
        confidence_threshold=.01,
        nms_iou_threshold=.0005,
        max_resolution=1080
    )
    
    src = load_source(opt.input_type, opt.input_file)
    
    i = 0
    while True:
        grabbed, frame, fname = get_frame(opt.input_type, opt.input_file, src, i)
        if not grabbed:
            break
        if opt.resize is not None:
            frame = resize_frame(frame, opt.resize)
        w, h = frame.shape[:2]
        if i == 0:
            writer = load_writer(opt.output_file_type, opt.output_file_loc, opt.input_file, (h,w))
        print('Processing frame ', i)
        t = time.time()
        boxes, scores = detect_faces(frame, detector)
        
        if opt.filter_boxes_scale is not None:
            boxes = filter_boxes_by_size(boxes, opt.filter_boxes_scale, (w,h))
        print(f"Detection time: {time.time()- t:.3f}")
        
        frame = anonymize_faces(boxes, frame, opt.anonymization_style)
        
        write_to_file(frame, opt.output_file_type, opt.output_file_loc, writer, fname)
        
        i+=1
        
    if opt.output_file_type == 'video':
        writer.release()
    
    print('Anonymized data stored at: ', opt.output_file_loc)
        
        
    
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_type', type=str, help='Whether input is video_file or image_folder')
    parser.add_argument('-input_file', type=str, help='Input video or image folder')
    parser.add_argument('--detector', type=str, default="RetinaNetResNet50", help='whether RetinaNet with Resnet or mobilenet backbone')
    parser.add_argument('--filter_boxes_scale', type=float, default=1/8, help='if not None, will filter out unlikely boxes that are larger than 1/x of the height and width of image')
    parser.add_argument('--resize', type=float, default=None, help='scale or size to resize to, if None, image is not resized')
    parser.add_argument('--output_file_type', type=str, default='image', help = 'whether to write anonymization to images or video')
    parser.add_argument('--output_file_loc', type = str, default ='output')
    parser.add_argument('--anonymization_style', type=str, default='blur')
    opt = parser.parse_args()
    anonymize_input(opt)
        