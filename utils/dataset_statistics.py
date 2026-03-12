import numpy as np
from typing import Any


def get_dataset_statistics(data_train: dict, model_opts: dict, 
                           use_precomputed_values: bool = False) -> dict:
    """ Get dataset statistics for various data features (mean, std dev, etc.). 
        Statistics are only computed on training data to avoid data leakage
    Args:
        data_train [dict]: training data dictionary
        model_opts [dict]: model options dictionary
        use_precomputed_values [bool]: if set to True, precomputed values will be used
                                       instead of recomputing values
    """
    if use_precomputed_values:
        if model_opts["dataset_full"] == "jaad_all":
            dataset_statistics = {
                "dataset_means": {
                    'scene_context_with_segmentation_v0': [0.020564525163150705],
                    'scene_context_with_segmentation_v5': [0.02057521288912929],
                    'scene_context': [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                    'scene_context_doubled': [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                    'scene_context_non_static': [0.43168591788409766, 0.41768097869166754, 0.43307793210699286],
                    'scene_context_with_ped_overlays_combined': [0.4301901984398352, 0.42690076535846205, 0.4316534206674368],
                    'local_box': [0.11011224069223947, 0.10042360925670331, 0.113358578575617],
                },
                "dataset_std_devs": {
                    'scene_context_with_segmentation_v0': [0.017205330101661215], 
                    'scene_context_with_segmentation_v5': [0.017224886418055325], 
                    'scene_context': [0.29391770977924025, 0.2938884989361296, 0.2861626088201237], 
                    'scene_context_doubled': [0.29391770977924025, 0.2938884989361296, 0.2861626088201237], 
                    'scene_context_non_static': [0.29414961750007607, 0.29414149452145005, 0.28640809656576754],
                    'scene_context_with_ped_overlays_combined': [0.2992303758909483, 0.3024261744979176, 0.2865064567576991],
                    'local_box': [0.15354079266033985, 0.14666361540451084, 0.1599767555115886] 
                }
            }
        elif model_opts["dataset_full"] == "pie":
            dataset_statistics = {
                "dataset_means": {
                    'scene_context_with_segmentation_v0': [0.019261059868129638], 
                    'scene_context_with_segmentation_v5': [0.019334776265065773], 
                    'scene_context_with_ped_overlays_combined': [0.440628720401945, 0.4292507260208848, 0.4038140097004474], 
                    'scene_context_with_ped_overlays_previous': [0.4362198059582853, 0.415231076887648, 0.40352444356608014], 
                    'scene_context_non_static': [0.43955526540739354, 0.41036402920270165, 0.40498522525493474],
                    'local_box': [0.0820003314644778, 0.0819983742481533, 0.08879805281323737]
                },
                "dataset_std_devs": {
                    'scene_context_with_segmentation_v0': [0.016403327185491264], 
                    'scene_context_with_segmentation_v5': [0.016449318402970814], 
                    'scene_context_with_ped_overlays_combined': [0.28405485551807624, 0.2639542459273181, 0.24111443759504633], 
                    'scene_context_with_ped_overlays_previous': [0.27642635889627565, 0.2513282281173903, 0.24122525908505713], 
                    'scene_context_non_static': [0.27419735941404255, 0.24653309211399166, 0.24074212125307612],
                    'local_box': [0.13943468852625082, 0.14117864963143234, 0.15075024342837487]
                }
            }
        else:
            raise Exception()
        """
        dataset_statistics = {
            "dataset_means": {
                "scene_context": [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                "local_context": [0.287280191104146, 0.2733837271647457, 0.3027266527874068],
            },
            "dataset_std_devs": {
                "scene_context": [0.29391770977924025, 0.2938884989361296, 0.2861626088201237],
                "local_context": [[0.14456805092887873, 0.14980047848430367, 0.15326681137702644]],
            }
        }
        """

    else:
        # Calculate mean and std dev of all images in dataset
        means, std_devs = {}, {}

        for item in data_train["data"][0]:
            for t_idx, data_type in enumerate(data_train["data_params"]["data_types"]):
                # Do not calculate statistics if data type is not image-like
                if data_type == "local_box":
                    _calculate_stats_for_img_like_data(data_type, means, std_devs, item, t_idx)
                elif (type(item) is tuple or len(item.shape) <= 3) and \
                    "context" not in data_type:
                    continue
                else:
                    _calculate_stats_for_img_like_data(data_type, means, std_devs, item, t_idx)

        dataset_means, dataset_std_devs = {}, {}
        for data_type in data_train["data_params"]["data_types"]:
            if not data_type in means:
                #print(f"WARNING: No statistics computed for data type {data_type}")
                continue

            means_t = np.asarray(means[data_type])
            dataset_means[data_type] = np.mean(means_t, axis=0).tolist()

            std_devs_t = np.asarray(std_devs[data_type])
            dataset_std_devs[data_type] = np.mean(std_devs_t, axis=0).tolist()

        dataset_statistics = {
            "dataset_means": dataset_means,
            "dataset_std_devs": dataset_std_devs
        }

    # Calculate statistics for trajectory features
    calculate_stats_for_trajectory_data(data_train["data"][0],
                                        data_train["data"][1],
                                        dataset_statistics,
                                        model_opts,
                                        use_precomputed_values=True)

    return dataset_statistics


def _calculate_stats_for_img_like_data(data_type: str, means: dict, 
                                       std_devs: dict, item: tuple, t_idx: int):
    if data_type not in means:
        means[data_type] = []
        std_devs[data_type] = []
    
    img = item[0][t_idx]
    img = _format_img(img)

    num_channels = img.shape[-1]
    mean = [np.mean(img[...,i]) for i in range(num_channels)]
    std_dev = [np.std(img[...,i]) for i in range(num_channels)]
    means[data_type].append(mean)
    std_devs[data_type].append(std_dev)


def calculate_stats_for_trajectory_data(data: Any, labels: np.ndarray, 
                                        dataset_statistics: dict, model_opts: dict,
                                        include_labels: bool = False,
                                        use_precomputed_values: bool = False,
                                        trajectory_overlays: bool = False):
    
    if use_precomputed_values:
        dataset_statistics["dataset_maxs"], dataset_statistics["dataset_mins"] = {}, {}

        # Get stats for 'box' + 'speed'
        # Here, statistics include labels (pred_len=60)
        if model_opts["dataset_full"] == "jaad_all":
            
            #dataset_statistics["dataset_means"]["trajectory"] = [-7.817597278751057, -2.797630704496746, 0.7966554592107592, 15.765265538056195, 0.46580601126806626] # 0.5881889890930191, 0.49563012446195803, 0.7114383036465168, 2.7137792721052323]
            #dataset_statistics["dataset_std_devs"]["trajectory"] = [161.24783689412033, 14.863155394437634, 161.40917654572766, 29.585598109148503, 0.2441041362407904] # 0.05964588498939262, 0.24599167728011406, 0.095895595316637, 1.350200146298904]
            #dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 126.0, 1864.0, 350.0, 0.9979166666666667] # 0.7694444444444445, 0.9994791666666667, 0.9990740740740741, 4.0
            #dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0] # 0.31203703703703706, 0.0078125, 0.4546296296296296, 0.0

            #if trajectory_overlays:
            dataset_statistics["dataset_means"]["trajectory"] = [-7.817597278751057, -2.797630704496746, 0.7966554592107592, 15.765265538056195, 2.4889659647260074]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [161.24783689412033, 14.863155394437634, 161.40917654572766, 29.585598109148503, 1.4469956924143323]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 126.0, 1864.0, 350.0, 4.0]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]

        elif model_opts["dataset_full"] == "jaad_beh":

            dataset_statistics["dataset_means"]["trajectory"] = [-17.27282171389908, -6.375631717282119, 0.8653072058708721, 30.30301819033492, 2.7640398678012854]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [225.56754977876653, 17.441062982192516, 225.14992865469864, 39.937830636813565, 1.2849233429522937]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1479.0, 86.0, 1620.0, 350.0, 4.0]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]
        
        elif model_opts["dataset_full"] == "pie":

            dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 6.55515245931165]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 9.758136061619542]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 54.00958464000001]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, 0.0]

        elif model_opts["dataset_full"] == "combined":

            dataset_statistics["dataset_means"]["trajectory"] = [-2.3904858567951655, -3.5280755481445425, 3.418520166310761, 11.067645944221596, 5.541632936654937]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [142.47869284250606, 15.787625933538196, 143.5664739120694, 25.60335049534862, 8.578877125881265]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 221.32, 1864.0, 350.0, 54.00958464000001]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -407.0899999999999, -1655.0, -217.10000000000002, 0.0]

    else: # compute statistics
        traj_np = []
        for item in data:
            traj_values = item[0][-1]
            traj_np.append(traj_values.tolist())
        traj_np = np.array(traj_np).squeeze(axis=1)
        traj_np = np.reshape(traj_np, (traj_np.shape[0]*traj_np.shape[1], traj_np.shape[-1]))

        if include_labels:
            labels_np = []
            for item in labels:
                labels_np.append(item.tolist())
            labels_np = np.array(labels_np)
            labels_np = np.reshape(labels_np, 
                                   (labels_np.shape[0]*labels_np.shape[1], labels_np.shape[-1]))
            traj_np = np.concatenate((traj_np, labels_np), axis=0)

        means = np.mean(traj_np, axis=0)
        std_devs = np.std(traj_np, axis=0)
        maxs = np.max(traj_np, axis=0)
        mins = np.min(traj_np, axis=0)

        dataset_statistics["dataset_means"]["trajectory"] = list(means)
        dataset_statistics["dataset_std_devs"]["trajectory"] = list(std_devs)
        dataset_statistics["dataset_maxs"], dataset_statistics["dataset_mins"] = {}, {}
        dataset_statistics["dataset_maxs"]["trajectory"] = list(maxs)
        dataset_statistics["dataset_mins"]["trajectory"] = list(mins)

    return

def _format_img(img: np.ndarray):
    img = img / 255 # Normalize to a value between 0 and 1
    if img.shape[0] == 1 and img.shape[1] == 1:
        img = np.squeeze(img, axis=(0,1))
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
    return img
