from scipy.ndimage import generate_binary_structure
from tqdm import tqdm

from models.hugging_face.utils.semantic_segmentation import SegformerForSemanticSegmentationWrapper
from utils.scene_graph.pedestrian import get_pedestrians_traffic_element, get_target_pedestrian_traffic_element
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
                    action_predict_ref=None,
                    trajectories=False):
    video_graph = False

    # Get list of semantic maps
    try:
        semantic_map_idx = model_opts["obs_input_type"].index("scene_video_with_segmentation_v0")
        video_graph = True
        semantic_maps = processed_data[semantic_map_idx]
        if trajectories:
            semantic_maps = data["trajectories_segm_maps"]
    except:
        try:
            if not get_previous_scene_graph:
                semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v0")
            else:
                semantic_map_idx = model_opts["obs_input_type"].index("scene_context_with_segmentation_v5")
            semantic_maps = processed_data[semantic_map_idx]
        except:
            video_graph = True
            semantic_maps, feat_shape = \
                action_predict_ref.get_context_data(model_opts, data, data_type, "scene_video_with_segmentation_v0")

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

    # scene_type = "scene_graph_doubled" if get_previous_scene_graph else "scene_graph"
    if model_opts["seq_type"] == "trajectory":
        scene_type = "scene_graph_v2"
    else:
        scene_type = "scene_graph" if get_previous_scene_graph else "scene_graph" # TODO: this is a temporary workaround
    path_to_features, _ = get_path(save_folder=scene_type,
                                   dataset=model_opts["dataset_full"],
                                   save_root_folder='data/features')
    feature_folder_path = os.path.join(path_to_features, data_type)
    print(f"Generating features type={scene_type}, save_path={feature_folder_path}")

    # Compute scene graphs
    for i in tqdm(range(len(semantic_maps))):
        img_id = semantic_maps[i][0].rsplit("/", 1)[-1].split(".")[0]
        if model_opts["seq_type"] == "trajectory":
            feature_save_path = os.path.join(feature_folder_path, f"{scene_type}_seq_{i}_{img_id}")
        else:
            feature_save_path = os.path.join(feature_folder_path, f"{scene_type}_seq_{i}")
        feature_save_path = f"{feature_save_path}.pkl" if not trajectories else f"{feature_save_path}_traj.pkl"
        if os.path.exists(feature_save_path):
            feature = open_pickle_file(feature_save_path)
            features.append(feature)
            continue

        if video_graph:
            features_for_seq = []
            for t in range(len(semantic_maps[i])):
                map_path = semantic_maps[i][t]
                sorted_occurences = get_scene_graph_for_timestep(
                                        map_path, data, i, t, MAP_SIZE,
                                        SEGFORMER_MODEL, model_opts, scene_context,
                                        trajectories=trajectories,
                                        debug=False
                                    )
                features_for_seq.append(sorted_occurences)
            features.append(features_for_seq)
            save_data_in_pkl(feature_folder_path, feature_save_path, features_for_seq)

        else:
            map_path = semantic_maps[i][0]
            sorted_occurences = get_scene_graph_for_timestep(
                                    map_path, data, i, t, MAP_SIZE,
                                    SEGFORMER_MODEL, model_opts, scene_context,
                                    debug=False
                                )
            feature = [sorted_occurences] * seq_len # copy features for all sequence idx
            features.append(feature)
            save_data_in_pkl(feature_folder_path, feature_save_path, feature)

    features = np.array(features)
    feat_size = features.shape[1:]

    return features, feat_size

def get_scene_graph_for_timestep(map_path, data, i, t, map_size,
                                 segformer_model, model_opts, scene_context,
                                 trajectories=False,
                                 debug=False):
    map = open_pickle_file(map_path)

    occurences = []

    occurences = get_target_pedestrian_traffic_element(
        data, i, t, map, map_size, occurences, model_opts, 
        trajectories=trajectories, debug=debug)

    occurences = get_pedestrians_traffic_element(
        data, i, t, segformer_model, map, scene_context, 
        map_size, occurences, 
        trajectories=trajectories, debug=debug)
    
    occurences = get_road_traffic_element(
        data, i, t, segformer_model, map, scene_context, 
        map_size, occurences, 
        trajectories=trajectories, debug=debug)
    
    occurences = get_sidewalk_traffic_element(
        data, i, t, segformer_model, map, scene_context, 
        map_size, occurences, 
        trajectories=trajectories, debug=debug)
    
    occurences = get_vehicle_traffic_element(
        data, i, t, segformer_model, map, scene_context, 
        map_size, occurences, 
        trajectories=trajectories, debug=debug)
    
    vertex_edges_indices = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1 is a vertex, 0 is an edge
    vertices = [e for i, e in enumerate(occurences) if vertex_edges_indices[i]==1]
    edges = [e for i, e in enumerate(occurences) if vertex_edges_indices[i]==0]
    others = [e for i, e in enumerate(occurences) if vertex_edges_indices[i]==2]
    sorted_occurences = vertices + edges + others

    return sorted_occurences
    
    
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
