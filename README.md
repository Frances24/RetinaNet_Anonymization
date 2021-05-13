# Face Detection in Pytorch with RetinaFace and 

- Lightweight single-shot face detection from the paper [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) adapted from https://github.com/biubug6/Pytorch_Retinaface.

## Getting started
Add the weights for the corresponding model to 'weights'folder and run
```
python anonymize_faces -input_type image_folder -input_file videos\\MOT1609 --output_file_type image
```
This will look for images in the input_file folder, and save the results with same name in output folder

## Simple API
To perform detection you can simple use the following lines:

```python
import cv2
import face_detection
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB
im = cv2.imread("path_to_im.jpg")[:, :, ::-1]

detections = detector.detect(im)
```

This will return a tensor with shape `[N, 5]`, where N is number of faces and the five elements are `[xmin, ymin, xmax, ymax, detection_confidence]`

### Batched inference

```python
import numpy as np
import face_detection
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# [batch size, height, width, 3]
images_dummy = np.zeros((2, 512, 512, 3))

detections = detector.batched_detect(im)
```


## References:

RetinaFace Inference Code from: https://github.com/hukkelas/DSFD-Pytorch-Inference, adapted from training and inference code at: https://github.com/biubug6/Pytorch_Retinaface



```

## Citation
If you find this code useful, remember to cite the original authors:
```

@inproceedings{deng2019retinaface,
  title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
  booktitle={arxiv},
  year={2019}

```
