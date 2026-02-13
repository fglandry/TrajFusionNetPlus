import copy
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from models.hugging_face.model_trainers.sambranch import load_pretrained_sam_branch
from models.hugging_face.model_trainers.vambranch import load_pretrained_vam_branch
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TorchTimeseriesDataset, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from models.hugging_face.utils.create_optimizer import get_optimizer
from models.custom_layers_pytorch import SelfAttention
from models.custom_layers_pytorch import SelfAttention
from utils.data_load import DataGenerator
from utils.action_predict_utils.run_in_subprocess import run_and_capture_model_path

NET_INNER_DIM = 512
NET_OUTER_DIM = 40
DROPOUT = 0.1


class TrajFusionNetPlus(HuggingFaceTimeSeriesModel):

    def train(self,
              data_train: dict,  
              data_val: DataGenerator,
              batch_size: int,
              epochs: int,
              model_path: str,   
              generator: bool = False,
              train_opts: dict = None,  
              dataset_statistics: dict = None,
              hyperparams: dict = None,
              class_w: list = None,
              test_only: bool = False,
              train_end_to_end: bool = False,
              submodels_paths: dict = None,
              *args, **kwargs):
        """ Train model
        Args:
            data_train [dict]: training data (data_train['data'][0] contains the generator)
            data_val [DataGenerator]: validation data
            batch_size [int]: batch size
            epochs [int]: number of epochs
            model_path [str]: path where the model will be saved
            generator [bool]: specifies if a generator is used to load data
            train_opts [str]: training options (includes learning rate)
            dataset_statistics [dict]: contains dataset statistics such as avg / std dev per feature
            hyperparams [dict]: dict containing hyperparameters to use, if enabled
            class_w [list]: class weights
            test_only [bool]: is set to True, model will not be trained, only tested
            train_end_to_end [bool]: if True, all modules in the network will be trained (see modular
                                     training section in the paper)
            submodels_paths [dict]: dictionary containing paths to submodels saved on disk
        """

        print("Starting model loading for model TrajFusionNetPlus ===========================")

        # Get hyperparameters (if specified) and model configs
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.class_w = class_w

        # If training end-to-end, start by training submodels
        if train_end_to_end:
            submodels_paths = train_submodels(
                dataset=kwargs["model_opts"]["dataset_full"],
                submodels_paths=submodels_paths)

        model = TrajFusionNetForClassification(config_for_huggingface,
                                               class_w=class_w,
                                               dataset_statistics=dataset_statistics,
                                               dataset_name=kwargs["model_opts"]["dataset_full"],
                                               submodels_paths=submodels_paths)
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
            get_image_transform=True, img_model_config=None,
            get_seg_maps_transforms = False if kwargs["model_opts"].get("skip_seg_maps_transforms") else True,
            dataset_statistics=dataset_statistics)

        warmup_ratio = 0.1
        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=train_opts["lr"],
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs = epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1
        )
        
        if test_only:
            optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts)
            trainer = self._get_trainer(model, args, train_dataset, 
                                        val_dataset, optimizer, lr_scheduler)
        else:
            # Train model
            print("Starting training of model TrajFusionNet ===========================")
            trainer = self.train_with_initial_vam_branch_disabling(
                model, epochs, args, train_dataset,
                val_dataset, data_train, train_opts
            )
     
        return {
            "trainer": trainer,
            "val_transform": val_transforms_dicts
        }
    
    def train_with_initial_vam_branch_disabling(self,
            model: Any, epochs: int, 
            args: TrainingArguments, train_dataset: TorchTimeseriesDataset, 
            val_dataset: TorchTimeseriesDataset, data_train: dict, train_opts: dict):
        
        best_metric = 0
        best_trainer = None
        half_epochs = round(epochs / 2)
        
        # Run first part of training procedure with the VAM branch disabled for 15 epochs
        # to improve learning in the SAM branch.
        # In order to do this, the weights in the VAM projection layer ('van_output_embed')
        # as well as the associated learning rate are set to zero
        with torch.no_grad(): 
            model.van_output_embed.weight.zero_()
            model.van_output_embed.bias.zero_()

        for i in range(half_epochs):
            
            # Get custom optimizer to set learning rate to zero in the VAM projection layer
            optimizer, lr_scheduler = get_optimizer(self, model, args, 
                train_dataset, val_dataset, data_train, train_opts,
                disable_vam_branch=True, nb_epochs_disabled=15, epoch_index=i+1)
            
            trainer = self._get_trainer(model, args, train_dataset, 
                                        val_dataset, optimizer, lr_scheduler)
            trainer.args.num_train_epochs = 1
            
            trainer.train()

            if trainer.state.best_metric > best_metric:
                best_trainer = trainer
                best_metric = trainer.state.best_metric

        # Run second part of training procedure with the VAM branch re-enabled       
        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts,
            disable_vam_branch=False) # learning rate of the VAM projection layer is
                                      # reset to the global learning rate

        trainer = self._get_trainer(model, args, train_dataset, 
                                    val_dataset, optimizer, lr_scheduler)
        trainer.args.num_train_epochs = half_epochs

        trainer.train()

        if trainer.state.best_metric > best_metric:
            best_trainer = trainer
            best_metric = trainer.state.best_metric if trainer.state.best_metric else 0

        return best_trainer

    def _get_trainer(self, model, args, train_dataset, 
                     val_dataset, optimizer, lr_scheduler):
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
            optimizers=(optimizer, lr_scheduler)
        )
        return trainer

    def test(self,
             test_data: tuple,
             training_result: dict,
             model_info: dict,
             *args,
             dataset_name: str = "",
             generator: bool = False,
             test_only: bool = False,
             **kwargs):
        
        print("Starting inference using trained model ===========================")

        if test_only:
            pretrained_model = load_pretrained_trajfusionnet(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class TrajFusionNetForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 class_w: list = None,
                 dataset_statistics: dict = None,
                 dataset_name: str = "",
                 submodels_paths: dict = None
        ):
        super().__init__(config_for_huggingface)
        self._device = get_device()
        self._dataset = dataset_name
        self.dataset_statistics = dataset_statistics
        self.class_w = torch.tensor(class_w).to(self._device) if class_w else None
        self.num_labels = config_for_huggingface.num_labels

        self.combine_branches_with_attention = False
        self.combine_vans_with_attention = False

        # MODEL PARAMETERS ==========================================

        # Classifier head parameters
        self.van_output_size = 256
        self.max_classifier_hidden_size = NET_OUTER_DIM
        self.max_classifier_hidden_size_van = 1024
        self.fc1_neurons = 2 * self.max_classifier_hidden_size
        self.fc2_neurons = NET_OUTER_DIM
        
        # 'van_sequential' contains the 4 sequential VAN image encoders
        self.van_sequential = load_pretrained_vam_branch(
            dataset_name,
            add_classification_head=False,
            submodels_paths=submodels_paths)

        # Get pretrained encoder transformer (at the end of VAM branch)
        self.traj_class_TF = load_pretrained_sam_branch(
            dataset_name,
            add_classification_head=False,
            submodels_paths=submodels_paths)

        # Classifier layers
        if self.combine_branches_with_attention:
            self.self_attention = SelfAttention(self.max_classifier_hidden_size)
        if self.combine_vans_with_attention:
            self.self_attention_van = SelfAttention(self.max_classifier_hidden_size_van)

        self.van_output_embed = nn.Linear(40, NET_OUTER_DIM)

        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc1 = nn.Linear(self.fc1_neurons, self.fc2_neurons)
        self.fc2 = nn.Linear(self.fc2_neurons, self.num_labels)

        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        timeseries_context: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        normalized_trajectory_values: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args, **kwargs
    ):
        """ Args:
        trajectory_values [torch.Tensor]: non-normalized observed trajectory values
            of shape [batch, seq_len, enc]
        normalized_trajectory_values [torch.Tensor]: normalized observed trajectory values
            of shape [batch, seq_len, enc]
        labels [torch.Tensor]: future target trajectory values of shape [batch, pred_len, enc]
            (between time t=0 and time t=60)
        """
        
        return_dict = self._on_entry(output_hidden_states, return_dict)

        # Apply VAM branch to video frames ==================================================

        van_seq_output = self.van_sequential(
            video_context=video_context
        )
        
        van_seq_output = self.van_output_embed(van_seq_output) # shape=[batch, 40]

        
        # Apply SAM branch to trajectory values (and GAM branch to graph encodings) =========

        outputs_pred = self.traj_class_TF(
            trajectory_values=trajectory_values,
            normalized_trajectory_values=normalized_trajectory_values,
            timeseries_context=timeseries_context
        )

        # Fuse SAM branch and VAM branch =====================================================

        if self.combine_branches_with_attention:
            tuple_to_concat = [outputs_pred, van_seq_output]
            original_x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, combined_fc_len]
            
            x = self._concatenate_with_attention(self.self_attention, 
                    original_x, tuple_to_concat, self.max_classifier_hidden_size)
        else:
            tuple_to_concat = [outputs_pred, van_seq_output]
            x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, 80]

        # Apply fully-connected layers
        outputs = self.dropout(nn.ReLU()(self.fc1(x)))
        logits = self.fc2(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict,
                            class_w=self.class_w)

    def _concatenate_with_attention(self, self_attention, original_x, 
                                    tuple_to_concat, max_concat_size):
        
        # concatenate outputs with padding
        tuple_to_concat = self._pad_tensors(tuple_to_concat, max_concat_size)
        x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, nb_models, max_concat_size]

        attention_ctx, attn = self_attention(x)
        attention_ctx = attention_ctx.squeeze(2)
        x = torch.cat((original_x, attention_ctx), dim=1)

        return x

    def _pad_tensors(self, tensors: list, max_size: int, pad_dim=1):
        for idx, output_tensor in enumerate(tensors):
            tensors[idx] = F.pad(output_tensor, pad=(0, max_size - output_tensor.shape[pad_dim], 0, 0)) # shape=[batch, fc_len]
            if len(tensors[idx].shape) <= 2:
                tensors[idx] = tensors[idx].unsqueeze(1) # shape=[batch, X, fc_len]
        return tensors
    
    def _on_entry(self, output_hidden_states, return_dict):
        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.return_dict = return_dict
        return return_dict

def train_submodels(dataset: str,
                    submodels_paths: dict):

    # SAM module ===============================================================
    
    # Train encoder transformer
    enc_tf_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/TrajectoryTransformerb.yaml", 
         "-d", dataset, "-j", submodels_paths['traj_tf_path']])

    # VAM module ===============================================================
    
    # Train VAN with image context at time t and predicted trajectory overlays
    van_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/VAN.yaml", 
         "-d", dataset, "-j", submodels_paths['traj_tf_path']])
    
    # Train VAN with image context at time t-15 and observed trajectory overlays
    van_prev_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/VAN_previous.yaml", 
         "-d", dataset, "-j", submodels_paths['traj_tf_path']])
    
    submodels_paths.update({
        "enc_tf_path": enc_tf_path,
        "van_path": van_path,
        "van_prev_path": van_prev_path
    })

    return submodels_paths

def load_pretrained_trajfusionnet(dataset_name: str):
    if dataset_name == "combined":
        checkpoint = "data/models/pie/TrajFusionNet/weights_trajfusionnet_pie"
        raise Exception()
    if dataset_name == "pie":
        checkpoint = "data/models/pie/TrajFusionNetPlus/weights_trajfusionnetplus_pie"
    elif dataset_name == "jaad_all":
        checkpoint = "data/models/jaad_all/TrajFusionNetPlus/weights_trajfusionnetplus_jaadall"
    elif dataset_name == "jaad_beh":
        checkpoint = "data/models/jaad_beh/TrajFusionNet/weights_trajfusionnet_jaadbeh"
        
    pretrained_model = TrajFusionNetForClassification.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
        dataset_name=dataset_name,
        class_w=None)
    
    # Make all layers untrainable
    for child in pretrained_model.children():
        for param in child.parameters():
            param.requires_grad = False
    return pretrained_model

    
def get_config_for_context_timeseries(encoder_input_size, seq_len, hyperparams, 
                                      add_graph_context_to_lstm=False):

    hyperparams = hyperparams.get("timeseries_tf", {})

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "graph_type": "scene_graph",
        "pred_len": 0, # for Timesblock
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": 2, #77, #6161 # 84 # encoder input size - default value,
        "d_model": 128, # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": DROPOUT, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 4), # num of heads
        "d_ff": hyperparams.get("d_ff", 512), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 2), # num of encoder layers (or 3)
        "seq_len": seq_len, # + 20, # input sequence length
        "num_class": NET_INNER_DIM, # number of neurons in last Linear layer at the end of model
        # ---------------------------------------------------------------------------------
        "label_len": 1, # Timesnet - start token length
        "num_kernels": 6, # Timesnet - for Inception
        "top_k": 5, # Timesnet - for TimesBlock
        "moving_avg": 3, # FEDformer - window size of moving average, default=25
        "dec_in": 7, # FEDformer - decoder input size
        "d_layers": 2, # FEDformer - num of decoder layers
        "c_out": 7, # FEDformer - output size
        "distil": True, # Informer - whether to use distilling in encoder, using this argument means not using distilling
        #"c_out": 77, # override - MICN - output size
        "p_hidden_dims": [128, 128], # Nonstationary transformer - hidden layer dimensions of projector (List)
        "p_hidden_layers": 2, # Nonstationary transformer - number of hidden layers in projector
        # "num_kernels": 3, # override - Pyraformer
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    time_series_dict_0 = copy.deepcopy(time_series_dict)
    time_series_dict_0["enc_in"] = encoder_input_size
    time_series_dict_0["task_name"] = "encoding"
    config_for_timeseries_lib_0 = TimeSeriesLibraryConfig(time_series_dict_0)
    return config_for_timeseries_lib_0, config_for_timeseries_lib
