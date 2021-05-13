import numpy as np
import torch
import typing
from abc import ABC, abstractmethod
from torchvision.ops import nms
from .box_utils import scale_boxes


def check_image(im: np.ndarray):
    assert im.dtype == np.uint8,\
        f"Expect image to have dtype np.uint8. Was: {im.dtype}"
    assert len(im.shape) == 4,\
        f"Expected image to have 4 dimensions. got: {im.shape}"
    assert im.shape[-1] == 3,\
        f"Expected image to be RGB, got: {im.shape[-1]} color channels"


class Detector(ABC):

    def __init__(
            self,
            confidence_threshold: float,
            nms_iou_threshold: float,
            device: torch.device,
            max_resolution: int,
            fp16_inference: bool,
            clip_boxes: bool):
        """
        Args:
            confidence_threshold (float): Threshold to filter out bounding boxes
            nms_iou_threshold (float): Intersection over union threshold for non-maxima threshold
            device ([type], optional): Defaults to cuda if cuda capable device is available.
            max_resolution (int, optional): Max image resolution to do inference to.
            fp16_inference: To use torch autocast for fp16 inference or not
            clip_boxes: To clip boxes within [0,1] to ensure no boxes are outside the image
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.max_resolution = max_resolution
        self.fp16_inference = fp16_inference
        self.clip_boxes = clip_boxes
        self.mean = np.array(
            [123, 117, 104], dtype=np.float32).reshape(1, 1, 1, 3)

    def detect(
            self, image: np.ndarray, shrink=1.0, prob_map=None) -> np.ndarray:
        """Takes an RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
        """
        image = image[None]
        if prob_map is None:
            boxes = self.batched_detect(image, shrink)
        else:
            boxes = self.batched_detect(image, shrink, prob_map)
        return boxes[0]

    @abstractmethod
    def _detect(self, image: torch.Tensor) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (torch.Tensor): shape [N, 3, height, width]
        Returns:
            torch.Tensor: of shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        raise NotImplementedError

    def filter_boxes(self, boxes: torch.Tensor, w, h) -> typing.List[np.ndarray]:
        """Performs NMS and score thresholding

        Args:
            boxes (torch.Tensor): shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        Returns:
            list: N np.ndarray of shape [B, 5]
        """
        final_output = []
        if not torch.is_tensor(boxes):
            boxes = torch.Tensor(boxes).to('cuda:0')
        for i in range(len(boxes)):
            scores = boxes[i, :,  4]
            keep_idx = scores >= self.confidence_threshold
            boxes_ = boxes[i, keep_idx, :-1]

            scores = scores[keep_idx]
            if scores.dim() == 0:
                final_output.append(torch.empty(0, 5))
                continue
            keep_idx = nms(boxes_, scores, self.nms_iou_threshold)
            scores = scores[keep_idx].view(-1, 1)
            boxes_ = boxes_[keep_idx].view(-1, 4)
            # print(output.size())
            keep_idx = []
            for i in range(len(boxes_)):
                if abs(boxes_[i,2]-boxes_[i,0]) < int(w/25) or abs(boxes_[i,3]- boxes_[i,1]) < int(h/25):
                    keep_idx.append(i)
                else:
                    continue
            boxes_ = boxes_[keep_idx]
            scores = scores[keep_idx]
            output = torch.cat((boxes_, scores), dim=-1)
            final_output.append(output)
        
        return final_output

    @torch.no_grad()
    def resize(self, image, shrink: float):
        if self.max_resolution is None and shrink == 1:
            return image
        height, width = image.shape[2:4]
        shrink_factor = self.max_resolution / max((height, width))
        if shrink_factor <= shrink:
            shrink = shrink_factor
        size = (int(height*shrink), int(width*shrink))
        image = torch.nn.functional.interpolate(image, size=size)
        return image

    def _pre_process(self, image: np.ndarray, shrink: float) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            torch.Tensor: shape [N, 3, height, width]
        """
        assert image.dtype == np.uint8
        height, width = image.shape[1:3]
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        image = self.resize(image, shrink)
        image = image.to(self.device)

        image = image.to(self.device)
        return image

    def _batched_detect(self, image: np.ndarray, prob_map=None) -> typing.List[np.ndarray]:
        boxes = self._detect(image)
        if prob_map is None:
            height, width = image.shape[1:3]
            boxes = self.filter_boxes(boxes, width, height)
        if self.clip_boxes:
            boxes = [box.clamp(0, 1) for box in boxes]
        return boxes
    
    def _account_for_prob_map(self, boxes, prob_map, h, w):
        if boxes[0].shape[0] > 0:
            for i in range(boxes[0].shape[0]):
                if abs(boxes[0][i,2]-boxes[0][i,0]) > int(prob_map.shape[0]/25) or abs(boxes[0][i,3]- boxes[0][i,1]) > int(prob_map.shape[1]/25):
                    continue
                x_min = max(0,int(boxes[0][i,0]))
                x_max = min(w, int(boxes[0][i,2]))
                y_min = max(0, int(boxes[0][i,1]))
                y_max = min(h, int(boxes[0][i,3]))
                boxes[0][i, 4] = min(1, boxes[0][i,4] + prob_map[y_min:y_max, x_min:x_max].sum())
                # print(prob_map[y_min:y_max, x_min:x_max].sum())
        return boxes
        
    @torch.no_grad()
    def batched_detect(
            self, image: np.ndarray, shrink=1.0, prob_map=None) -> typing.List[np.ndarray]:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            np.ndarray: a list with N set of bounding boxes of
                shape [B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        check_image(image)
        height, width = image.shape[1:3]
        image = self._pre_process(image, shrink)
        boxes = self._batched_detect(image, prob_map)
        boxes = [scale_boxes((height, width), box).cpu().numpy() for box in boxes]
        if prob_map is not None:
            boxes = self._account_for_prob_map(boxes, prob_map,  height, width)
            boxes = self.filter_boxes(boxes, width, height)
            boxes[0] = boxes[0].cpu().numpy()
        self.validate_detections(boxes)
        return boxes

    def validate_detections(self, boxes: typing.List[np.ndarray]):
        for box in boxes:
            assert np.all(box[:, 4] <= 1) and np.all(box[:, 4] >= 0),\
                f"Confidence values not valid: {box}"
