import copy
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, label as scipy_label

from utils.scene_graph.utils import get_ped_coord
from utils.semantic_processing import _get_distance_between_points, _get_angle_between_points
from utils.semantic_occurences import _is_veh_on_road
from utils.utils import *

struct = generate_binary_structure(2,2)


def get_vehicle_traffic_element(data, i, t, model, map, scene_context,
                                map_size, occurences, 
                                graphormer_encoding=True,
                                trajectories=False,
                                debug=False):
    veh_idx = 13
    
    """
    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        model.display_segmentation_map(display_map, img, unique_label_to_show=ped_idx)
    """

    occurences = _get_occurences_of_vehicles(data, i, t, model, map, 
                    scene_context, map_size, veh_idx, occurences, 
                    graphormer_encoding, 
                    trajectories=trajectories,
                    debug=debug)

    return occurences

def _get_occurences_of_vehicles(data, i, t, model, map, scene_context,
                                map_size, veh_idx, occurences, 
                                graphormer_encoding,
                                trajectories=False,
                                debug=False):
    
    MAX_DIST = int(math.hypot(map.shape[0], map.shape[1]))
    min_coord = [-1, -1]
    min_angle = math.pi
    ped_coord, bb_nb_pixels = get_ped_coord(data, i, t, map_size,
                                            trajectories=trajectories)
        
    # Get vehicles mask
    map = copy.deepcopy(map)
    map[map != veh_idx] = -1
    map[map == veh_idx] = 1
    map[map == -1] = 0

    # Erode vehicle pixels
    map = ndimage.binary_erosion(map, structure=struct, iterations=3).astype(map.dtype)

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
        #if nb_pixels > 2000:
        #    continue
        if cm[0] <= 80 or cm[0] >= 200: # ensure x-value is not too high or too low (likely a false positive)
            continue
        groups.append({
            "nb_pixels": nb_pixels,
            "cm": [round(cm[0]), round(cm[1])],
            "current_ped_dist": _get_distance_between_points(ped_coord, cm),
            "current_ped_angle": _get_angle_between_points(ped_coord, cm)
        })

    # Return features for all vehicle occurences encountered
    NB_GROUPS_TO_KEEP = 2
    groups_features = []
    #kept_groups.sort(key=lambda x: x["nb_pixels"], reverse=True)
    groups.sort(key=lambda x: x["current_ped_dist"])
    nb_groups = len(groups)
    nb_vehicles = nb_groups
    kept_groups = groups[0:min(NB_GROUPS_TO_KEEP, nb_groups)]
    nb_empty_groups = NB_GROUPS_TO_KEEP - nb_groups if nb_groups < NB_GROUPS_TO_KEEP else 0
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
            ])
        else:
            groups_features.extend([
                [g["cm"][0]/map_size, 
                 g["cm"][1]/map_size], # vertex6/7_veh_cm_coord
                [g["current_ped_dist"]/MAX_DIST, 
                 g["current_ped_angle"]/math.pi] # edge06/7_veh_cm_dist
            ])
    # return default occurence values when no vehicle is found
    for _ in range(nb_empty_groups):
        if not graphormer_encoding:
            groups_features.extend([
                min_coord[0]/map_size, # center of mass x
                min_coord[1]/map_size, # center of mass y
                MAX_DIST/MAX_DIST, # current veh dist
                min_angle/math.pi, # current veh angle,
            ])
        else:
            groups_features.extend([
                [min_coord[0]/map_size, 
                 min_coord[1]/map_size], # vertex6/7_veh_cm_coord
                [MAX_DIST/MAX_DIST, 
                 min_angle/math.pi] # edge06/7_ped_cm_dist
            ])

    # Get center of mass of largest vehicle, and if that vehicle
    # is located on the road
    groups.sort(key=lambda x: x["nb_pixels"], reverse=True)
    if len(groups) > 0:
        cm_largest_veh = groups[0]["cm"]
        veh_on_road = _is_veh_on_road(cm_largest_veh, map)
    else:
        veh_on_road = False
    veh_on_road = int(veh_on_road)
    
    #occurences.extend([nb_vehicles, veh_on_road]) # global occurence features
    occurences.extend(groups_features) # scene graph groups features
    return occurences
