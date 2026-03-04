from utils.utils import *

def _verify_if_patch_is_positive(coord, map, patch_size=10):
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
    nb_road_pixels = np.count_nonzero(patch > 0)
    if (nb_road_pixels / (patch_size*patch_size)) >= 0.3:
        return True
    else:
        return False

def get_ped_coord(data, i, t, 
                  map_size=224,
                  trajectories=False):
    if trajectories:
        bb = data["trajectories_normalized_abs"][i][t]
    else:
        bb = data["normalized_abs_box_org"][i][t+1]
        if "normalized_abs_box" in data:
            assert np.array_equal(
                data["normalized_abs_box"][i][t],
                data["normalized_abs_box_org"][i][t+1]
            )
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