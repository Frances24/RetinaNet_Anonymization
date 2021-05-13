
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:51:29 2021

@author: fran-
"""

import argparse
from det_objects import FaceDetectionObject, LoadSource, LoadWriter
from tools import run_detector, run_tracker, run_ensemble_detectors, anonymize_faces
import face_detection
import os


from config import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import time


cfg_map = {
    "RetinaNetResNet50": 'RETINANET_RES',
    "RetinaNetMobileNetV1": 'RETINANET_MOB',
    "DSFDDetector": 'DSFD',
    }

def write_to_file(faces, img, w_obj):
    w_obj.write_next(faces, img)
    
    return

def anonymize_input(opt):
    faces = FaceDetectionObject(opt)
    end = False
    ls_obj = LoadSource(faces)
    w_obj = LoadWriter(faces)
    cfg = get_config(opt.config)
    
    if not os.path.isdir(opt.output_file_loc):
        os.mkdir(opt.output_file_loc)
        
    if opt.add_tracking is True:
            cfg_tracking = cfg['DEEPSORT']
            deepsort = DeepSort(cfg_tracking['REID_CKPT'],
                            max_dist=cfg_tracking['MAX_DIST'], min_confidence=cfg_tracking['MIN_CONFIDENCE'],
                            nms_max_overlap=cfg_tracking['NMS_MAX_OVERLAP'], max_iou_distance=cfg_tracking['MAX_IOU_DISTANCE'],
                            max_age=cfg_tracking['MAX_AGE'], n_init=cfg_tracking['N_INIT'], nn_budget=cfg_tracking['NN_BUDGET'],
                            use_cuda=True)
            
            # deepsort = Sort(max_age = 20, min_hits=1, iou_threshold=0.001)
            
         
    while not end:
        img, grabbed = ls_obj.next(faces)
        if not grabbed:
            end = True
        start = time.time()
        if opt.ensemble is False:
            det_cfg=cfg[cfg_map[opt.detector[0]]]
            detector = face_detection.build_detector(opt.detector[0],
                    confidence_threshold= det_cfg['CONF_THRESH'], 
                    nms_iou_threshold=det_cfg['NMS_THRESH'],
                    max_resolution=det_cfg['MAX_RES']
                )
            run_detector(img, faces, detector)
            if end is True:
                break
            if opt.add_tracking is True:
                run_tracker(faces, img, deepsort, opt)
        else:
            detectors = []
            for det in opt.detector:
                det_cfg = cfg[cfg_map[det]]
                detector = face_detection.build_detector(det,
                                confidence_threshold= det_cfg['CONF_THRESH'],
                                nms_iou_threshold=det_cfg['NMS_THRESH'],
                                max_resolution=det_cfg['MAX_RES']
                            )
                detectors.append(detector)
            run_ensemble_detectors(img, faces, detectors)
        img = anonymize_faces(faces,img)
        write_to_file(faces, img, w_obj)
        print('Processing time: ', time.time() - start)
            
    w_obj.close(faces)
    print('\a')
    print('anonymization complete, file saved to: ', faces.output_file_loc)
    return
            

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help= 'Configuration file for detector parameters')
    parser.add_argument('--input-type', type=str, default='image_folder', help='Whether input is video_file or image_folder')
    parser.add_argument('--input', type=str, default='../../April21/ExtractedFrames/overlapping_time_sequences/2cam_24jun/Keyframes_C1460_20200624-135500', help='Input video or image folder')
    parser.add_argument('--start_from', type=int, default=0, help='Whether to start from a certain frame or image')
    parser.add_argument('--ensemble', type=bool, default=False, help='whether to use ensemble of models for detections')
    parser.add_argument('--ensemble_method', type=str, default='consensus', help='whether to use consensus, affirmative or uanimous ensembling')
    parser.add_argument('--detector', type=list, default=["RetinaNetResNet50"], help='source')
    parser.add_argument('--filter_boxes_scale', type=float, default=1/25, help='if not None, will filter out unlikely boxes that are larger than 1/x of the height and width of image')
    parser.add_argument('--add_tracking', type=bool, default=False, help='Whether to include sort tracking')
    parser.add_argument('--img-resize', type=float, default=0.5, help='scale to resize by')
    parser.add_argument('--sample_fps', type=int, default=0, help = 'downsample video file according to certain fps, 0=no downsample')
    parser.add_argument('--end_after', type=int, default=None, help='whether to end after a certain number of frames, None means continue to end')
    parser.add_argument('--output_file_type', type=str, default='image', help = 'whether to write anonymization to images or video')
    parser.add_argument('--output_file_loc', type = str, default ='../../April21/ExtractedFrames/overlapping_time_sequences/2cam_24jun/Keyframes_C1460_20200624-135500_anon')
    parser.add_argument('--manual_review', type=bool, default=False, help='Whether to perform review and add corrected detections')
    parser.add_argument('--blur_for_manual_review', type=bool, default=True, help= 'Whether to blur during review')
    parser.add_argument('--manual_review_skip', type=int, default=1, help='how many frames to skip when doing manual review, 1 means no skip')
    parser.add_argument('--anonymization_style', type=str, default='blur')
    opt = parser.parse_args()
    anonymize_input(opt)
    #videos/C1042_20200829-165500.mp4
   