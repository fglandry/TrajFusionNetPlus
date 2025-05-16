from scipy import ndimage

from models.hugging_face.utils.semantic_segmentation import SegformerForSemanticSegmentationWrapper
from utils.utils import *

def get_segm_map_features(data, processed_data, model_opts, debug=False):
    semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v0")
    semantic_maps = processed_data[semantic_map_idx]

    if debug:
        scene_context_idx = model_opts["obs_input_type"].index("scene_context")
        scene_context = processed_data[scene_context_idx]

    seq_len = data["box_org"].shape[1]
    MAP_SIZE = 224
    SEGFORMER_MODEL = SegformerForSemanticSegmentationWrapper()
    id2label = SEGFORMER_MODEL.model.config.id2label

    features = []

    for i in range(len(semantic_maps)):
        map_path = semantic_maps[i][0]
        map = open_pickle_file(map_path)
        if debug:
            img_path = scene_context[i][0]
            img = open_pickle_file(img_path)
            SEGFORMER_MODEL.display_segmentation_map(map, img)
            #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/scene_context_{str(time.time()).replace('.', '_')}_{str(i)}.png", img)
            
        ped_coord = _get_seg_map_scaled_ped_coords(data, i, MAP_SIZE)

        cm_dist, cm_angle, min_dist, is_ped_on_road, ped_sem_category = \
            get_distances_between_pedestrian_and_road(ped_coord, map)

        # Get distances using horizontal strip extracted above/under
        # the pedestrian on the segmentation map
        strip_x1 = ped_coord[0] - 30
        strip_x2 = ped_coord[0] + 30
        strip_x1 = 0 if strip_x1 < 0 else strip_x1
        strip_x2 = MAP_SIZE if strip_x2 > MAP_SIZE else strip_x2
        horizontal_strip = map[strip_x1:strip_x2, :]
        horizontal_strip_ped_coord = [30, ped_coord[1]]
        cm_dist_strip, cm_angle_strip, min_dist_strip, is_ped_on_road_strip, ped_sem_category_strip = \
            get_distances_between_pedestrian_and_road(
                horizontal_strip_ped_coord, horizontal_strip, is_hor_strip=True)
        
        feature = [[cm_dist, cm_angle, min_dist,
                    cm_dist_strip, cm_angle_strip, min_dist_strip, 
                    is_ped_on_road_strip]] \
                        * seq_len # copy features for all sequence idx
        features.append(feature)
    
    features = np.array(features)
    return features, features.shape[1:]

def get_pixels_of_traffic_elements(data, processed_data, model_opts, debug=False):
    semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v0")
    semantic_maps = processed_data[semantic_map_idx]

    if debug:
        scene_context_idx = model_opts["obs_input_type"].index("scene_context")
        scene_context = processed_data[scene_context_idx]

    seq_len = data["box_org"].shape[1]
    MAP_SIZE = 224
    SEGFORMER_MODEL = SegformerForSemanticSegmentationWrapper()
    id2label = SEGFORMER_MODEL.model.config.id2label

    features = []

    #useful_categories = ["sidewalk", "pole", "traffic light", "traffic sign", "person", "car"]
    useful_categories = ["pole", "traffic light", "traffic sign"]
    useful_category_indexes = _get_useful_category_indexes(useful_categories, id2label)

    for i in range(len(semantic_maps)):
        map_path = semantic_maps[i][0]
        map = open_pickle_file(map_path)
        
        if debug:
            img_path = scene_context[i][0]
            img = open_pickle_file(img_path)
            SEGFORMER_MODEL.display_segmentation_map(map, img)
            #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/scene_context_{str(time.time()).replace('.', '_')}_{str(i)}.png", img)

        nb_pixels_per_category = []
        # 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light'
        # 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car'
        # 'truck', 'bus', 'train', 'motorcycle', 'bicycle'

        for id, label in id2label.items():
            nb_px = np.count_nonzero(map==id) // 30
            nb_pixels_per_category.append(nb_px)
            #if label == "traffic sign":
                #if nb_px_traffic_sign >= 1:
                #    SEGFORMER_MODEL.display_segmentation_map(map, img, unique_label_to_show=id)

        useful_nb_pixels_per_category = [nb_pixels_per_category[i] for i in useful_category_indexes]
        #useful_nb_pixels_per_category = [1 if p>=1 else 0 for p in useful_nb_pixels_per_category]

        feature = [useful_nb_pixels_per_category] * seq_len # copy features for all sequence idx
        features.append(feature)
    
    features = np.array(features)
    return features, features.shape[1:]

def _get_useful_category_indexes(useful_categories, id2label):
    useful_category_indexes = []
    for category in useful_categories:
        category_idx = [id for id in id2label if id2label[id]==category][0]
        useful_category_indexes.append(category_idx)
    useful_category_indexes.sort()
    return useful_category_indexes

def _get_seg_map_scaled_ped_coords(data, i, t, map_size,
                                   trajectories=False):
    """ Get pedestrian coordinates on 224 x 224 segmentation map
    """
    if trajectories:
        ped_box_coords = data["trajectories_org"][i][t]
    else:
        ped_box_coords = data["box_org"][i][t]
    ped_coord = [
        int((ped_box_coords[1]+ped_box_coords[3]) / 2 * (map_size / 1080)),
        int((ped_box_coords[0]+ped_box_coords[2]) / 2 * (map_size / 1920))
    ]
    if ped_coord[0] > map_size or ped_coord[1] > map_size:
        raise Exception
    return ped_coord

def get_distances_between_pedestrian_and_road(
        ped_coord: list, map: np.ndarray, is_hor_strip=False):

    MAX_DIST = int(math.hypot(map.shape[0], map.shape[1]))
    min_dist = MAX_DIST
    cm_dist = MAX_DIST
    min_coord = [-1, -1]
    min_angle = math.pi
    
    road_cm = _get_road_center_of_mass(map)
    cm_dist = _get_distance_between_points(ped_coord, road_cm)
    cm_angle = _get_angle_between_points(ped_coord, road_cm)
    is_ped_on_road = int(_verify_if_patch_is_road(ped_coord, map, patch_size=20))
    ped_sem_category = _get_patch_category(ped_coord, map)

    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x][y] == 0: # road
                dist = _get_distance_between_points(ped_coord, [x, y])
                if dist < min_dist and _verify_if_patch_is_road([x, y], map):
                    min_dist = dist
                    min_coord = [x, y]
                    # min_angle = _get_angle_between_points(org_ped_coord, [x, y])

    return cm_dist, cm_angle, min_dist, is_ped_on_road, ped_sem_category

def _verify_if_patch_is_road(coord, map, patch_size=10, road_idx=0):
    patch_coords = [coord[0] - patch_size/2, # x1
                    coord[0] + patch_size/2, # x2
                    coord[1] - patch_size/2, # y1
                    coord[1] + patch_size/2] # y2
    for c in patch_coords:
        if c < 0:
            c = 0
        if c >= map.shape[0]:
            c = map.shape[0] - 1
    
    patch = map[int(patch_coords[0]):int(patch_coords[1]), 
                int(patch_coords[2]):int(patch_coords[3])]
    nb_road_pixels = np.count_nonzero(patch == road_idx)
    if (nb_road_pixels / (patch_size*patch_size)) >= 0.3:
        return True
    else:
        return False
    
def _get_patch_category(coord, map, patch_size=30):
    patch_coords = [coord[0] - patch_size/2, # x1
                    coord[0] + patch_size/2, # x2
                    coord[1] - patch_size/2, # y1
                    coord[1] + patch_size/2] # y2
    for idx, c in enumerate(patch_coords):
        if c < 0:
            patch_coords[idx] = 0
        elif idx in [0,1]: # x values
            if c >= map.shape[0]:
                patch_coords[idx] = map.shape[0] - 1
        elif idx in [2,3]: # y values
            if c >= map.shape[1]:
                patch_coords[idx] = map.shape[1] - 1
    
    patch = map[int(patch_coords[0]):int(patch_coords[1]), 
                int(patch_coords[2]):int(patch_coords[3])]
    
    # In some cases, there could be errors in the bbox coordinates 
    # (from the dataset). In this case, return 19 as category index.
    if patch.size == 0:
        return 19
    
    u, c = np.unique(patch, return_counts=True)
    category = u[c.argmax()]
    if category == 11: # don't return 'person' as category
        category = u[_get_idx_of_second_max_in_list(c)] 

    return category

def _get_idx_of_second_max_in_list(_list):
    max = -1
    second_max = _list[0]
    idx_max = 0
    idx_second_max = 0
    for idx, elem in enumerate(_list):
        if elem > max:
            idx_second_max = idx_max
            second_max = max
            idx_max = idx
            max = elem
        elif elem > second_max:
            idx_second_max = idx
            second_max = elem
    return idx_second_max

def _get_road_center_of_mass(map: np.ndarray, road_idx: int = 0) -> list:
    road_area = (map == road_idx).astype(int)
    if not np.any(road_area):
        return [int(map.shape[0]/2), int(map.shape[1]/2)] # return mid-point
    cm = ndimage.measurements.center_of_mass(road_area)
    return [int(cm[0]), int(cm[1])]

def _get_distance_between_points(pointA: list, pointB: list):
    dist = int(math.dist(pointA, pointB))
    return dist

def _get_angle_between_points(pointA: list, pointB: list) -> float:
    dx = pointA[0] - pointB[0] # A.x - B.x due to orientation of cartesian plan on image
    dy = pointB[1] - pointA[1] # B.y - A.y due to orientation of cartesian plan on image
    angle = math.atan2(dx, dy)
    return angle
