from scipy.ndimage import generate_binary_structure

from models.hugging_face.utils.semantic_segmentation import SegformerForSemanticSegmentationWrapper
from utils.scene_graph.pedestrian import get_pedestrians_traffic_element
from utils.scene_graph.road import get_road_traffic_element
from utils.scene_graph.sidewalk import get_sidewalk_traffic_element
from utils.scene_graph.vehicle import get_vehicle_traffic_element
from utils.utils import *

struct = generate_binary_structure(2,2)

"""
def get_previous_scene_graph(data, processed_data, model_opts, debug=False):

    current_scene_graph_data = processed_data[-1]
    for idx in current_scene_graph_data.shape[0]:

    test = 10

def _past_element_contains_same_video_id(self, index, debug=True):
    previous_index = index - 1 if index - 1 >= 0 else 0
    current_video_id = self.data.data[0][index][0].split("/")[-2]
    previous_video_id = self.data.data[0][previous_index][0].split("/")[-2]

    if debug:
        img1 = open_pickle_file(self.data.data[0][previous_index][0])
        img2 = open_pickle_file(self.data.data[0][index][0])

        if not self.segformer_model:
            self.segformer_model = SegformerForSemanticSegmentationWrapper()
        self.segformer_model.display_segmentation_map(img1, None, get_img_combined_with_segmentation_map=False)
        self.segformer_model.display_segmentation_map(img2, None, get_img_combined_with_segmentation_map=False)

    return current_video_id == previous_video_id
"""

def get_scene_graph(data, processed_data, model_opts,
                    data_type="train",
                    get_previous_scene_graph=False, 
                    debug=False,
                    format_for_graphormer=True):
    if not get_previous_scene_graph:
        semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v0")
    else:
        semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v5")
    
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

    scene_type = "scene_graph_doubled" if get_previous_scene_graph else "scene_graph"
    path_to_features, _ = get_path(save_folder=scene_type,
                               dataset=model_opts["dataset"],
                               save_root_folder='data/features')
    feature_folder_path = os.path.join(path_to_features, data_type)
    print(f"Generating features type={scene_type}, save_path={feature_folder_path}")

    for i in range(len(semantic_maps)):
        feature_save_path = os.path.join(feature_folder_path, f"{scene_type}_seq_{i}.pkl")
        if os.path.exists(feature_save_path):
            feature = open_pickle_file(feature_save_path)
            features.append(feature)
            continue

        map_path = semantic_maps[i][0]
        map = open_pickle_file(map_path)
        """
        if i==200:
            scene_context_path = scene_context[i][0]
            scene_img = open_pickle_file(scene_context_path)
            SEGFORMER_MODEL.display_segmentation_map(map, scene_img, get_img_combined_with_segmentation_map=True)
        debug = False # ToDo: remove
        """

        occurences = []

        """
        occurences = get_target_pedestrian_traffic_element(
            data, i, occurences, model_opts, debug)
        """
        
        occurences = get_road_traffic_element(
            data, i, SEGFORMER_MODEL, map, scene_context, 
            MAP_SIZE, occurences, debug=debug)
        
        occurences = get_sidewalk_traffic_element(
            data, i, SEGFORMER_MODEL, map, scene_context, 
            MAP_SIZE, occurences, debug=debug)
        
        occurences = get_pedestrians_traffic_element(
            data, i, SEGFORMER_MODEL, map, scene_context, 
            MAP_SIZE, occurences, debug=debug)
        
        occurences = get_vehicle_traffic_element(
            data, i, SEGFORMER_MODEL, map, scene_context, 
            MAP_SIZE, occurences, debug=debug)
        
        vertex_edges_indices = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1 is a vertex, 0 is an edge
        # vertex_edges_indices = [1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]
        vertices = [e for i, e in enumerate(occurences) if vertex_edges_indices[i]==1]
        edges = [e for i, e in enumerate(occurences) if vertex_edges_indices[i]==0]
        others = [e for i, e in enumerate(occurences) if vertex_edges_indices[i]==2]
        sorted_occurences = vertices + edges + others
        
        feature = [sorted_occurences] * seq_len # copy features for all sequence idx
        features.append(feature)

        # Save the file
        save_data_in_pkl(feature_folder_path, feature_save_path, feature)

    #if format_for_graphormer:
    #    features = _format_data_for_graphormer(features)

    features = np.array(features)
    feat_size = features.shape[1:]

    return features, feat_size

def _format_data_for_graphormer(features):
    test = 10

    for i in range(len(features)):
        for seq_idx in range(len(features[0])):
            features[i][seq_idx] = {
                "input_nodes": None, # torch.LongTensor,
                "input_edges": None, # torch.LongTensor,
                "attn_bias": None, # torch.Tensor,
                "in_degree": None, # torch.LongTensor,
                "out_degree": None, # torch.LongTensor,
                "spatial_pos": None, # torch.LongTensor,
                "attn_edge_type": None # torch.LongTensor,
            }
    return features


def get_target_pedestrian_traffic_element(data, i, occurences, model_opts, 
                                          debug=False):
    features = []

    for d_type in model_opts['obs_input_type']:
        if "box" in d_type: # ToDo: verify if that always holds
            last_seq_element = data[d_type][i][-1]
            features.extend(last_seq_element.tolist())

    occurences.extend(features)
    return occurences
