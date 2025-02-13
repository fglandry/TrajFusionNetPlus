import copy
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, label as scipy_label

from utils.scene_graph.utils import _get_ped_coord, _verify_if_patch_is_positive
from utils.semantic_processing import _get_distance_between_points, \
    _get_seg_map_scaled_ped_coords, _get_angle_between_points
from utils.utils import *

struct = generate_binary_structure(2,2)


def get_pedestrians_traffic_element(data, i, model, map, scene_context,
                                    map_size, occurences, 
                                    graphormer_encoding=True,
                                    debug=False):
    ped_idx = 11
    
    """
    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        model.display_segmentation_map(display_map, img, unique_label_to_show=ped_idx)
    """

    occurences = _get_occurences_of_pedestrians(
                    data, i, model, map, scene_context, map_size, 
                    ped_idx, occurences, graphormer_encoding, debug)

    return occurences

def _get_occurences_of_pedestrians(data, i, model, map, scene_context,
                                   map_size, ped_idx, occurences, 
                                   graphormer_encoding,
                                   debug=False):
    
    MAX_DIST = int(math.hypot(map.shape[0], map.shape[1]))
    min_coord = [-1, -1]
    min_angle = math.pi
    ped_coord, bb_nb_pixels = _get_ped_coord(data, i, map_size)
        
    # Get pedestrians mask
    map = copy.deepcopy(map)
    map[map != ped_idx] = -1
    map[map == ped_idx] = 1
    map[map == -1] = 0

    # Dilate pedestrian pixels
    map = ndimage.binary_dilation(map, structure=struct, iterations=3).astype(map.dtype)

    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        model.display_segmentation_map(display_map, img, unique_label_to_show=1)

    labeled_array, num_features = scipy_label(map, structure=struct)
    num_small_features = 0
    
    groups = []
    for i in range(num_features):
        group_idx = i + 1
        nb_pixels = np.count_nonzero(labeled_array==group_idx)
        cm = ndimage.measurements.center_of_mass(labeled_array==group_idx)
        if nb_pixels <= 100:
            num_small_features = num_small_features + 1
            continue
        if nb_pixels > 2000:
            continue
        if cm[0] <= 80 or cm[0] >= 200: # ensure x-value is not too high or too low (likely a false positive)
            continue
        groups.append({
            "nb_pixels": nb_pixels,
            "cm": [round(cm[0]), round(cm[1])],
            "current_ped_dist": _get_distance_between_points(ped_coord, cm),
            "current_ped_angle": _get_angle_between_points(ped_coord, cm),
            "is_current_ped": False
        })  

    # Verify if groups correspond to the current pedestrian
    # bounding box, and if the current pedestrian is in a pedestrian group
    in_group = False
    for g in groups:
        if g["current_ped_dist"] < 40:
            g["is_current_ped"] = True
        elif g["current_ped_dist"] < 80:
            ratio_of_pedestrian_bbs = bb_nb_pixels / g["nb_pixels"]
            if ratio_of_pedestrian_bbs > 0.5 and ratio_of_pedestrian_bbs < 2:
                in_group = True
    in_group = int(in_group)

    # Return features for all pedestrian occurences encountered
    NB_GROUPS_TO_KEEP = 2
    groups_features = []
    kept_groups = [g for g in groups if not g["is_current_ped"]]
    kept_groups.sort(key=lambda x: x["nb_pixels"], reverse=True)
    #kept_groups.sort(key=lambda x: x["current_ped_dist"])
    nb_kept_groups = len(kept_groups)
    nb_pedestrians = nb_kept_groups 
    kept_groups = kept_groups[0:min(NB_GROUPS_TO_KEEP, nb_kept_groups)]
    nb_empty_groups = NB_GROUPS_TO_KEEP - nb_kept_groups if nb_kept_groups < NB_GROUPS_TO_KEEP else 0
    if len(kept_groups) == 1:
        test = 10
    if len(kept_groups) == 2:
        test = 10
    for g in kept_groups:
        if not graphormer_encoding:
            groups_features.extend([
                g["cm"][0]/map_size,
                g["cm"][1]/map_size,
                g["current_ped_dist"]/MAX_DIST,
                g["current_ped_angle"]/math.pi,
                #g["nb_pixels"]
            ])
        else:
            groups_features.extend([
                [g["cm"][0]/map_size,
                 g["cm"][1]/map_size], # vertex4/5_ped_cm_coord
                [g["current_ped_dist"]/MAX_DIST, 
                 g["current_ped_angle"]/math.pi] # edge04/5_ped_cm_dist
            ])
    # return default occurence values when no pedestrian is found
    for _ in range(nb_empty_groups):
        if not graphormer_encoding:
            groups_features.extend([
                min_coord[0]/map_size, # center of mass x
                min_coord[1]/map_size, # center of mass y
                MAX_DIST/MAX_DIST, # current ped dist
                min_angle/math.pi, # current ped angle,
                #0 # nb_pixels
            ])
        else:
            groups_features.extend([
                [min_coord[0]/map_size, 
                 min_coord[1]/map_size], # vertex4/5_ped_cm_coord
                [MAX_DIST/MAX_DIST, 
                 min_angle/math.pi] # edge04/5_ped_cm_dist
            ])

    #occurences.extend([in_group, nb_pedestrians]) # global occurence features
    occurences.extend(groups_features) # scene graph groups features
    return occurences
