import copy
from scipy import ndimage
from scipy.ndimage import label as scipy_label, generate_binary_structure

from models.hugging_face.utils.semantic_segmentation import SegformerForSemanticSegmentationWrapper
from utils.semantic_processing import _get_distance_between_points, _get_idx_of_second_max_in_list
from utils.utils import *

struct = generate_binary_structure(2,2)
#a = np.ones((5,5))
#s = a.tolist()
#s = [[1,1,1,1],
#     [1,1,1,1],
#     [1,1,1,1],
#     [1,1,1,1]
#    ]


def get_occurences_of_traffic_elements(data, processed_data, model_opts, debug=False):
    semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v0")
    semantic_maps = processed_data[semantic_map_idx]

    if debug:
        scene_context_idx = model_opts["obs_input_type"].index("scene_context")
        scene_context = processed_data[scene_context_idx]
    else:
        scene_context = None

    seq_len = data["box_org"].shape[1]
    MAP_SIZE = 224
    SEGFORMER_MODEL = SegformerForSemanticSegmentationWrapper()
    #id2label = SEGFORMER_MODEL.model.config.id2label

    features = []

    for i in range(len(semantic_maps)):
        map_path = semantic_maps[i][0]
        map = open_pickle_file(map_path)

        occurences = []
        """
        occurences = _get_occurences_of_pedestrians(
            data, i, SEGFORMER_MODEL, map, scene_context, MAP_SIZE, 
            occurences, seq_len, debug)
        """

        occurences = _get_occurences_of_vehicles(
            i, SEGFORMER_MODEL, map, scene_context, MAP_SIZE, 
            occurences, debug)
        
        feature = [occurences] * seq_len # copy features for all sequence idx
        features.append(feature)
    
    features = np.array(features)
    return features, features.shape[1:]

def _get_occurences_of_pedestrians(data, i, t, model, map, scene_context,
                                   map_size, occurences, debug=False):
    ped_idx = 11
    ped_coord, bb_nb_pixels = get_ped_coord(data, i, t, map_size)
        
    # Get pedestrians mask
    map[map != ped_idx] = -1
    map[map == ped_idx] = 1
    map[map == -1] = 0

    # Dilate pedestrian pixels
    map = ndimage.binary_dilation(map, structure=struct, iterations=3).astype(map.dtype)

    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        display_map[display_map == 1] = 11
        model.display_segmentation_map(display_map, img, unique_label_to_show=ped_idx)
        #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/scene_context_{str(time.time()).replace('.', '_')}_{str(i)}.png", img)

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
        #nm_neighbors = _get_number_of_neighbors(labeled_array, i+1)
        groups.append({
            "nb_pixels": nb_pixels,
            "cm": [round(cm[0]), round(cm[1])]
        })

    nb_pedestrians = len(groups)

    # Verify if at least one group corresponds to the current pedestrian
    # bounding box, and if the current pedestrian is in a group
    found, in_group = False, False
    for g in groups:
        dist = _get_distance_between_points(ped_coord, g["cm"])
        if dist < 40:
            found = True
        elif dist < 80:
            ratio_of_pedestrian_bbs = bb_nb_pixels / g["nb_pixels"]
            if ratio_of_pedestrian_bbs > 0.5 and ratio_of_pedestrian_bbs < 2:
                in_group = True

    if not found:
        nb_pedestrians = nb_pedestrians + 1
    in_group = int(in_group)

    occurences.extend([nb_pedestrians, in_group])
    return occurences

def _get_occurences_of_vehicles(i, model, map, scene_context,
                                map_size, occurences, debug=False):
    veh_idx = 13
        
    # Get vehicles mask
    org_map = copy.deepcopy(map)
    map[map != veh_idx] = -1
    map[map == veh_idx] = 1
    map[map == -1] = 0

    # Erode pedestrian pixels
    map = ndimage.binary_erosion(map, structure=struct, iterations=3).astype(map.dtype)

    if debug:
        img_path = scene_context[i][0]
        img = open_pickle_file(img_path)
        display_map = copy.deepcopy(map)
        display_map[display_map == 1] = 11 # color in yellow
        model.display_segmentation_map(display_map, img, unique_label_to_show=11)
        #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/scene_context_{str(time.time()).replace('.', '_')}_{str(i)}.png", img)

    labeled_array, num_features = scipy_label(map, structure=struct)
    num_small_features = 0
    
    groups = []
    for idx in range(num_features):
        group_idx = idx + 1
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
            "cm": [round(cm[0]), round(cm[1])]
        })

    nb_vehicles = len(groups)
    
    # Get center of mass of largest vehicle, and if that vehicle
    # is located on the road
    groups.sort(key=lambda x: x["nb_pixels"], reverse=True)
    if len(groups) > 0:
        cm_largest_veh = groups[0]["cm"]
        cm_largest_veh_norm = [cm_largest_veh[0]/map_size, cm_largest_veh[1]/map_size]
        veh_on_road = _is_veh_on_road(cm_largest_veh, org_map)
    else:
        cm_largest_veh_norm = [0, 0]
        veh_on_road = False
    veh_on_road = int(veh_on_road)

    occurences.extend([nb_vehicles, cm_largest_veh_norm[0], cm_largest_veh_norm[1], veh_on_road])
    return occurences

def get_ped_coord(data, i, t, map_size=224):
    bb = data["normalized_abs_box"][i][t]
    bb_nb_pixels = round(abs((bb[3] - bb[1]) * (bb[2] - bb[0]) * map_size * map_size))
    coord = [round((bb[1] + bb[3])*map_size/2), round((bb[0] + bb[2])*map_size/2)]
    return coord, bb_nb_pixels

def _is_veh_on_road(coord, map, patch_size=50):
    patch_coords = [coord[0], # x1 - only take bottom half of patch
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
    # (from the dataset). In this case, return True.
    if patch.size == 0:
        return True
    
    u, c = np.unique(patch, return_counts=True)

    # Remove 'vehicle' and 'sky' from categories
    idxs_to_remove = []
    for u_idx, u_elem in enumerate(u):
        if u_elem in [10, 13]:
            idxs_to_remove.append(u_idx)
    u = np.delete(u, idxs_to_remove)
    c = np.delete(c, idxs_to_remove)

    if u.size == 0: # it is likely that the vehicle is very large and located on the road
        return True

    category1 = u[c.argmax()]
    #category2 = u[_get_idx_of_second_max_in_list(c)]

    if 0 not in [category1]: # 0 is the index for road
        return False
    return True
