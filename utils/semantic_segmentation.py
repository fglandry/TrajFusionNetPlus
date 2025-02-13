import numpy as np
from PIL import Image

from models.hugging_face.utils.semantic_segmentation import DeepLabV3ForSemanticSegmentationWrapper
from models.hugging_face.utils.semantic_segmentation import SegformerForSemanticSegmentationWrapper

from utils.utils import *


def get_semantic_segmentation(img_features: np.ndarray,
                              img_data: np.ndarray,
                              target_dim: tuple, 
                              feature_type: str,
                              b: np.ndarray = None,
                              crop_mode: str = None,
                              compute_time=False,
                              use_segformer=True,
                              use_deeplabv3=False,
                              debug=True):
    """ 
    Args:
        img_features: image features after initial processing
        img_data: original image data
    """
    try:
        if use_segformer:
            SEGFORMER_MODEL = SegformerForSemanticSegmentationWrapper()
            output, class_idx_tsr, image = SEGFORMER_MODEL.run(img_features)

            if feature_type == 'ped_scene_segmentation':
                mask = get_pedestrian_segmentation_mask(b, img_data, output, crop_mode, 
                                                        target_dim, SEGFORMER_MODEL)
            elif 'scene_context_with_segmentation' in feature_type:
                mask = add_segmentation_map_to_img_features(SEGFORMER_MODEL, img_features, output, 
                                                            feature_type, class_idx_tsr, image)
                #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", mask[..., 0:3])
        elif use_deeplabv3:
            DEEPLABV3_MODEL = DeepLabV3ForSemanticSegmentationWrapper(
                compute_time=compute_time)
            _, _, _ = DEEPLABV3_MODEL.run(img_features)
            # TODO: code this part
            mask = None

        """
        if debug:
            #show_img_fcn(img_features)
            #show_img_fcn(output)
            cv2.imwrite("ped_orig.png", img_data) 
            cv2.imwrite("ped_output.png", output)
            test = 10
        """
        
    except:
        if feature_type == 'ped_scene_segmentation':
            mask = np.zeros((16, 16), dtype=np.uint8) # todo: revisit
        else:
            raise

    return mask


def add_segmentation_map_to_img_features(seg_model, img_features, segm_data, feature_type,
                                         class_idx_tsr, image):   
    if 'segmentation_v0' in feature_type:
        img_features = segm_data # simply return segmentation map
    elif 'segmentation_v2' in feature_type:
        img_features[..., 0] = segm_data # replace first RGB channel
    elif 'segmentation_v3' in feature_type:
        # Get combination of img and segmentation map (0.5 and 0.5 weights)
        img_features, _, _ = seg_model.get_img_combined_with_segmentation_map(class_idx_tsr, image)
        #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", img_features)
    elif 'segmentation_v4' in feature_type:
        # Get segmentation map only using 3 channels
        img_features, _, _ = seg_model.get_img_combined_with_segmentation_map(class_idx_tsr, image,
                                                                              img_weight=0.0, seg_weight=1.0)
    elif 'segmentation_v5' in feature_type:
        img_features = segm_data # return segmentation map, but first element in sequence instead of last (already taken care of in 'get_static_context_data')
    else:
        segm_data = np.expand_dims(segm_data, 2)    
        img_features = np.append(img_features, segm_data, axis=2)
    #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", img_features[..., 0]*4)
    return img_features


def get_pedestrian_segmentation_mask(b, img_data, output, crop_mode, 
                                     target_dim, seg_model, debug=False):
    # Resize bounding box coordinates from original dims to target dims
    b[0] = int(b[0] / img_data.shape[1] * output.shape[1])
    b[1] = int(b[1] / img_data.shape[0] * output.shape[0])
    b[2] = int(b[2] / img_data.shape[1] * output.shape[1])
    b[3] = int(b[3] / img_data.shape[0] * output.shape[0])

    # to get output, crop out bounding box around pedestrian
    output = crop_bbox(output, b, crop_mode, target_dim, skip_padding=True) # numpy ndarray for output
    class_idx_tsr = crop_bbox(class_idx_tsr, b, crop_mode, target_dim, skip_padding=True) # torch tensor

    if debug:
        try:
            # for display purposes, crop out bounding box around pedestrian
            image = Image.fromarray(crop_bbox(np.array(image), b, crop_mode, target_dim, skip_padding=True))
            seg_model.display_segmentation_map(class_idx_tsr, image)
        except:
            pass
    
    ped_mask = seg_model.get_pedestrian_mask(class_idx_tsr)
    return ped_mask
