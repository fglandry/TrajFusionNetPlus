import copy
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, label as scipy_label

from utils.scene_graph.utils import _verify_if_patch_is_positive
from utils.semantic_processing import _get_distance_between_points, \
    _get_seg_map_scaled_ped_coords, _get_angle_between_points
from utils.utils import *

struct = generate_binary_structure(2,2)


def get_sidewalk_traffic_element(data, i, model, map, scene_context,
                                 map_size, occurences, 
                                 graphormer_encoding=True,
                                 debug=False):
    sidewalk_idx = 1
        
    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        model.display_segmentation_map(display_map, img, unique_label_to_show=sidewalk_idx)

    ped_coord = _get_seg_map_scaled_ped_coords(data, i, map_size)

    occurences = _get_occurences_of_sidewalks(
                    i, model, map, scene_context,
                    map_size, ped_coord, occurences, 
                    graphormer_encoding, debug)

    return occurences

def _get_occurences_of_sidewalks(i, model, map, scene_context,
                                 map_size, ped_coord, occurences, 
                                 graphormer_encoding,
                                 debug=False):
    sidewalk_idx = 1
        
    # Get sidewalk mask
    map = copy.deepcopy(map)
    map[map != sidewalk_idx] = -1
    map[map == sidewalk_idx] = 1
    map[map == -1] = 0

    # Erode pedestrian pixels
    # map = ndimage.binary_erosion(map, structure=struct, iterations=3).astype(map.dtype)

    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        model.display_segmentation_map(display_map, img, unique_label_to_show=sidewalk_idx)

    # labeled_array, num_features = scipy_label(map, structure=struct)

    sidewalk_min_coord, sidewalk_min_dist, sidewalk_min_angle = \
        _get_distances_between_pedestrian_and_sidewalk(
            ped_coord, map)

    if not graphormer_encoding:
        occurences.extend([sidewalk_min_coord[0], 
                           sidewalk_min_coord[1], 
                           sidewalk_min_dist, 
                           sidewalk_min_angle])
    else:
        occurences.extend([
            [sidewalk_min_coord[0], 
             sidewalk_min_coord[1]], # vertex3_sidewalk_min_coord
            [sidewalk_min_dist, 
             sidewalk_min_angle] # edge03_sidewalk_min_dist
        ])
    return occurences

def _get_distances_between_pedestrian_and_sidewalk(
        ped_coord: list, map: np.ndarray, map_size=224):

    MAX_DIST = int(math.hypot(map.shape[0], map.shape[1]))
    sidewalk_min_dist = MAX_DIST
    sidewalk_min_coord = [-1, -1]
    sidewalk_min_angle = math.pi
    
    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x][y] > 0: # sidewalk
                dist = _get_distance_between_points(ped_coord, [x, y])
                if dist < sidewalk_min_dist and _verify_if_patch_is_positive([x, y], map):
                    sidewalk_min_dist = dist
                    sidewalk_min_coord = [x, y]
                    sidewalk_min_angle = _get_angle_between_points(ped_coord, sidewalk_min_coord)

    # Normalize coordinates
    sidewalk_min_coord = [sidewalk_min_coord[0]/map_size, 
                          sidewalk_min_coord[1]/map_size]
    sidewalk_min_dist = sidewalk_min_dist / MAX_DIST
    sidewalk_min_angle = sidewalk_min_angle / math.pi

    # sidewalk_min_coord: position of closest sidewalk pixel 
    # sidewalk_min_dist: distance between pedestrian and closest sidewalk pixel
    # sidewalk_min_angle: angle between pedestrian and closest sidewalk pixel               
    return sidewalk_min_coord, sidewalk_min_dist, sidewalk_min_angle
