import copy
from scipy.ndimage import generate_binary_structure

#from utils.feature_descriptors import get_hu_moments
from utils.semantic_processing import _get_distance_between_points, _get_seg_map_scaled_ped_coords, \
      _get_road_center_of_mass, _get_angle_between_points, _verify_if_patch_is_road, \
      _get_patch_category
from utils.utils import *

struct = generate_binary_structure(2,2)

def get_road_traffic_element(data, i, t, model, map, scene_context,
                             map_size, occurences,
                             graphormer_encoding=True,
                             trajectories=False,
                             debug=False):
    road_idx = 0
        
    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        model.display_segmentation_map(display_map, img, unique_label_to_show=road_idx)

    ped_coord = _get_seg_map_scaled_ped_coords(data, i, t, map_size,
                                               trajectories=trajectories)

    road_cm_coord, road_cm_dist, road_cm_angle, road_min_coord, \
        road_min_dist, road_min_angle, ped_sem_category = \
        get_distances_between_pedestrian_and_road(ped_coord, map, map_size)

    if not graphormer_encoding:
        # (x, y, d, theta)
        # 1. road center of mass
        # 2. closest road pixel
        occurences.extend([
            road_cm_coord[0], road_cm_coord[1], road_cm_dist, road_cm_angle, 
            road_min_coord[0], road_min_coord[1], road_min_dist, road_min_angle,
            ped_sem_category
        ])
    else:
        occurences.extend([                             # vertex0_central_ped
            # [normalized_ped_coord[0], normalized_ped_coord[1]],
            [road_cm_coord[0], road_cm_coord[1]],    # vertex1_road_cm
            [road_cm_dist, road_cm_angle],           # edge01_road_cm_dist
            [road_min_coord[0], road_min_coord[1]],  # vertex2_min_coord
            [road_min_dist, road_min_angle],          # edge02_road_min_dist
            # [ped_sem_category, ped_sem_category]
        ])

    return occurences

def get_distances_between_pedestrian_and_road(
        ped_coord: list, map: np.ndarray, map_size=224):

    MAX_DIST = int(math.hypot(map.shape[0], map.shape[1]))
    road_min_dist = MAX_DIST
    road_cm_dist = MAX_DIST
    road_min_coord = [-1, -1]
    road_min_angle = math.pi
    
    road_cm_coord = _get_road_center_of_mass(map)
    road_cm_dist = _get_distance_between_points(ped_coord, road_cm_coord)
    road_cm_angle = _get_angle_between_points(ped_coord, road_cm_coord)
    #is_ped_on_road = int(_verify_if_patch_is_road(ped_coord, map, patch_size=20))
    ped_sem_category = _get_patch_category(ped_coord, map)

    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x][y] == 0: # road
                dist = _get_distance_between_points(ped_coord, [x, y])
                if dist < road_min_dist and _verify_if_patch_is_road([x, y], map):
                    road_min_dist = dist
                    road_min_coord = [x, y]
                    road_min_angle = _get_angle_between_points(ped_coord, road_min_coord)

    # Normalize coordinates
    road_cm_coord = [road_cm_coord[0]/map_size, road_cm_coord[1]/map_size]
    road_min_coord = [road_min_coord[0]/map_size, road_min_coord[1]/map_size]
    road_cm_dist = road_cm_dist / MAX_DIST
    road_min_dist = road_min_dist / MAX_DIST
    road_cm_angle = road_cm_angle / math.pi
    road_min_angle = road_min_angle / math.pi

    # road_cm_coord: position of road center of mass 
    # road_cm_dist: distance between pedestrian and road center of mass
    # road_cm_angle: angle between pedestrian and road center of mass
    # road_min_coord: position of closest road pixel 
    # road_min_dist: distance between pedestrian and closest road pixel
    # road_min_angle: angle between pedestrian and closest road pixel
    # ped_sem_category: where is the pedestrian located?            
    return road_cm_coord, road_cm_dist, road_cm_angle, \
           road_min_coord, road_min_dist, road_min_angle, ped_sem_category

"""
def _get_hu_moments_from_road_segm_map(img, category_idx):
    input_img = np.copy(img)

    # Prepare image for thresholding
    input_img[input_img != category_idx] = -1
    input_img[input_img == category_idx] = 255
    input_img[input_img == -1] = 0

    moments = get_hu_moments(input_img)
    moments = [[m, m] for m in moments]

    return moments
"""
