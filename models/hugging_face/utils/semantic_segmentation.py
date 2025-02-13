import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import time
import torch
from torch import nn
from typing import Union

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor
from transformers import AutoModelForSemanticSegmentation

from utils.utils import *

SWIN2SR_MODEL = None


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class SegformerForSemanticSegmentationWrapper(metaclass=Singleton):
    
    def __init__(self):
        # "nvidia/segformer-b0-finetuned-ade-512-512" -> for initial testing
        # "matei-dorian/segformer-b5-finetuned-human-parsing" -> results are not that great, especially for smaller pedestrians
        # "mattmdjaga/segformer_b2_clothes" -> potentially even worse than the previous one
        # "nvidia/segformer-b3-finetuned-cityscapes-1024-1024" -> sometimes very useful, but most times it is not
        self.processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
        self.extractor = SegformerFeatureExtractor()

    def run(self, img_features: np.ndarray, debug=False):
        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        #image = Image.open(requests.get(url, stream=True).raw)
        #show_image(img_features)
        img_features = cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB) # is the model input really RGB?
        image = Image.fromarray(img_features)
        #image.show()

        inputs = self.processor(images=image, return_tensors="pt")

        outputs = self.model(**inputs)

        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        class_idx_tsr = self.get_segmentation_map(logits, image)

        if debug:
            #list(logits.shape)
            self.display_segmentation_map(class_idx_tsr, image)

        class_idx_img_np = class_idx_tsr.numpy()
        #class_idx_np = np.expand_dims(class_idx_tsr.numpy(), axis=2)
        #cv2.imshow("image", class_idx_np)
        return class_idx_img_np, class_idx_tsr, image

    def get_segmentation_map(self, logits: torch.Tensor, image: Image.Image) -> torch.Tensor:
        #self.processor.post_process_semantic_segmentation()

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                        size=image.size[::-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)
        
        # Second, apply argmax on the class dimension
        seg = upsampled_logits.argmax(dim=1)[0]
        return seg

    def get_img_combined_with_segmentation_map(self, seg: torch.Tensor, image: Image.Image,
                                               unique_label_to_show=-1,
                                               img_weight=0.5, seg_weight=0.5):
        
        color_seg = self._convert_map_into_displayable_img(seg, unique_label_to_show)

        # Show image + mask
        org_img = np.array(image)
        org_color_seg = np.array(color_seg)
        img = np.array(image) * img_weight + color_seg * seg_weight
        img = img.astype(np.uint8)
        #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", np.squeeze(img))
        return img, org_img, org_color_seg

    def _convert_map_into_displayable_img(self, seg, unique_label_to_show=-1):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            if unique_label_to_show != -1:
                if label == unique_label_to_show:
                    color_seg[seg == label, :] = np.array([255,255,0]) # color
                else:
                    color_seg[seg == label, :] = np.array([255,255,255])
            else:
                color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        return color_seg

    def display_segmentation_map(self, seg: Union[torch.Tensor, np.array], 
                                 org_img: Union[Image.Image, np.array],
                                 unique_label_to_show=-1,
                                 get_img_combined_with_segmentation_map=True):

        if get_img_combined_with_segmentation_map:
            if not isinstance(org_img, Image.Image):
                image = Image.fromarray(org_img)
            else:
                image = org_img
            img, org_img, org_color_seg = self.get_img_combined_with_segmentation_map(
                seg, image, unique_label_to_show=unique_label_to_show, 
                img_weight=0.7, seg_weight=0.3)

        #plt.figure(figsize=(15, 10))
        #plt.imshow(org_img)
        #plt.show()
        #plt.imshow(org_color_seg)
        #plt.show()
        #plt.imshow(img)
        #plt.show()
        #plt.savefig(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png")
        #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/org_output_{str(time.time()).replace('.', '_')}.png", org_img)  

        # Get obtained classes
        map = np.array(seg) if not isinstance(seg, np.ndarray) else seg
        classes_map = np.unique(map).tolist()
        unique_classes = {self.model.config.id2label[idx]: idx \
                          if idx!=255 else None for idx in classes_map}
        #print("Classes in this image:", unique_classes)

        # Get number of pixel segmented per class
        classes_idx_map = unique_classes.copy()
        unique, counts = np.unique(map, return_counts=True)
        idx_counts_map = dict(zip(unique, counts))
        classes_counts_map = {a: idx_counts_map[b] for a, b in classes_idx_map.items()}
        sorted_classes_counts_map = {k: v for k, v in \
                                     sorted(classes_counts_map.items(), \
                                            key=lambda item: item[1],
                                            reverse=True)}
        print("Number of pixels segmented per class:", sorted_classes_counts_map)
        #if "rider" in sorted_classes_counts_map:
        #    cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", img)  

        nb_pixels_traffic_sign = sorted_classes_counts_map.get("traffic sign")
        nb_pixels_traffic_light = sorted_classes_counts_map.get("traffic light")
        nb_pixels_road = sorted_classes_counts_map.get("road")
        nb_pixels_car = sorted_classes_counts_map.get("car")
        nb_pixels_sidewalk = sorted_classes_counts_map.get("sidewalk")
        nb_pixels_person = sorted_classes_counts_map.get("person")

        #if (nb_pixels_road and nb_pixels_road > 400):
        if get_img_combined_with_segmentation_map:
            cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", img)  
            #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/img_output_{str(time.time()).replace('.', '_')}.png", org_img)  
        else:
            color_seg = np.array(self._convert_map_into_displayable_img(map))
            cv2.imwrite(f"/home/francois/MASTER/sem_imgs/img_output_{str(time.time()).replace('.', '_')}.png", color_seg)  


    def get_pedestrian_mask(self, seg: torch.Tensor,
                            debug=True) -> np.ndarray:

        person_idx = [k for k, v in self.model.config.id2label.items() \
                      if (v == "person" or v == "rider")][0]
        
        color_seg = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8) # height, width, 3
        color_seg[seg == person_idx] = 255 # palette[person_idx]

        _, mask = cv2.threshold(color_seg, 128, 255, cv2.THRESH_BINARY)

        if debug:
            cv2.imwrite(f"/home/francois/MASTER/sem_imgs/mask_{str(time.time()).replace('.', '_')}.png", mask)   
        
        return mask

class DeepLabV3ForSemanticSegmentationWrapper(metaclass=Singleton):
    
    def __init__(self, compute_time=False):
        self.processor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
        self.compute_time = compute_time
        if self.compute_time:
            print("Started calculation of average time spent per DeepLabV3 segmentation computation")
            self.start_time = time.time()
            self.num_samples = 0


    def run(self, img_features: np.ndarray, debug=False):
        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        #image = Image.open(requests.get(url, stream=True).raw)
        #show_image(img_features)
        img_features = cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB) # is the model input really RGB?
        image = Image.fromarray(img_features)
        #image.show()
        start_time = time.time()
        inputs = self.processor(images=image, return_tensors="pt")

        outputs = self.model(**inputs)
        predicted_mask = self.processor.post_process_semantic_segmentation(outputs)

        if self.compute_time:
            self.num_samples += 1
            total_time = time.time() - start_time
            print(f"Time to compute 1 image: {total_time}")

        # TODO: code this part
        return None, None, None

    def get_calc_time(self):
        total_time = time.time() - self.start_time
        avg_time = total_time / self.num_samples
        print(f"Average time per segmentation computation: {avg_time}")
        return avg_time

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]
