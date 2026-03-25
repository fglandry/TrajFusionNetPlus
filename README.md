# TrajFusionNet+

This repository contains the code for the paper **TrajFusionNet+: Transformer-Based Prediction of Pedestrian Crossing Intention via Fusion of Trajectory Representations and Scene Graphs**

TrajFusionNet+ is a transformer-based model for pedestrian crossing intention prediction. The architecture builds upon the previous TrajFusionNet model and combines sequential and visual representations of pedestrian trajectory with a graph-based representation of the scene context.

<img src="docs/architecture.png" alt="TrajFusionNet Architecture" width="500">

TrajFusionNet+ is composed of three branches:
- Sequence Attention Module (SAM)
  - Processes a sequential representation of past and predicted pedestrian trajectories.
- Visual Attention Module (VAM)
  - Utilizes a visual representation of pedestrian trajectories by overlaying observed and predicted bounding boxes onto scene images.
- Graph Attention Module (GAM)
  - Extracts pedestrian-centric graphs from segmented scene images and captures non-Euclidean spatial relationships between traffic elements.

## Set up

Start by creating a conda environment:

```bash
conda create -n trajfusionnet-env python=3.10 pytorch "torchvision<0.15" pytorchvideo pytorch-cuda accelerate tensorflow -c pytorch -c nvidia -c conda-forge
```

The pytorch-cuda version might need to be specified depending on your NVIDIA driver version.

Then, install the remaining libraries with pip:

```bash
pip install -r requirements.txt
```

### Downloading the datasets and model weights

The PIE and JAAD datasets need to be downloaded and processed by following instructions provided in the following GitHub repos: [https://github.com/aras62/PIE](https://github.com/aras62/PIE) and [https://github.com/ykotseruba/JAAD](https://github.com/ykotseruba/JAAD)

After downloading the datasets, export environment variables pointing to the datasets' locations:
```bash
export JAAD_PATH=<jaad_dataset_location>
export PIE_PATH=<pie_dataset_location>
```

The model weights are downloaded automatically from HuggingFace when running the code ([https://huggingface.co/efl7126/trajfusionnet-plus](https://huggingface.co/efl7126/trajfusionnet-plus)). Alternatively, the model weights can be downloaded from Dropbox: [https://www.dropbox.com/scl/fo/b66bmekyms6wiuynmfx5c/AH-tzUh3a3YlhVHxsiQzb1o?rlkey=ddjqsun0wghsq8fvn9jrapet0&st=lfo9hu91&dl=0](https://www.dropbox.com/scl/fo/b66bmekyms6wiuynmfx5c/AH-tzUh3a3YlhVHxsiQzb1o?rlkey=ddjqsun0wghsq8fvn9jrapet0&st=lfo9hu91&dl=0). You will have to extract the zip file to the data/ directory by running:
```bash
unzip <download_location>/weights.zip -d data/
```

## Inference

To perform model inference, execute the following command:
```bash
python3 train_test.py -c config_files/TrajFusionNetPlusInference.yaml --test_only
```

The dataset to use and other config parameters can be modified in `config_files/TrajFusionNetPlusInference.yaml`

## Training

To train the model end-to-end, run:
```bash
python3 train_test.py -c config_files/TrajFusionNetPlus.yaml --train_end_to_end
```

## Citation

If you find this repo useful, please cite the following publication:

TBD

Depending on your use of the code, please also cite the following:

* A large part of the codebase was forked from [https://github.com/ykotseruba/PedestrianActionBenchmark](https://github.com/ykotseruba/PedestrianActionBenchmark)

* The code for the transformer submodels comes from the TSLib library [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)


## Authors

<!--
* Francois-Guillaume Landry
* Moulay Akhloufi

Please email efl7126@umoncton.ca (FG Landry) or create an issue if you experience problems with running the code or setting up the environment.
-->

## License

This project is licensed under the MIT License
