import copy
from typing import Optional

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from libs.time_series_library.models_tsl.Tokengt import Model as TokengtTransformer
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from models.custom_layers_pytorch import CrossAttention
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from utils.data_load import DataGenerator

PRED_LEN = 60
NET_INNER_DIM = 512
DROPOUT = 0.1


class GAMBranch(HuggingFaceTimeSeriesModel):

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
              test_only: bool = False,
              *args, **kwargs):
        """ Train model
        Args:
            data_train [dict]: training data (data_train['data'][0] contains the generator)
            data_val [DataGenerator]: validation data
            model_path [str]: path where the model will be saved
            train_opts [str]: training options (includes learning rate)
            dataset_statistics [dict]: contains dataset statistics such as avg / std dev per feature
            hyperparams [dict]: hyperparameters to change during training
            test_only [bool]: is set to True, model will not be trained, only tested
        """
        
        print("Starting model loading for model GAM Branch (Graph Attention Module) ======================")

        # Get parameters to be used by TSLib library
        data_element = data_train['data'][0][0][0][0]
        encoder_input_size = data_element.shape[-1]
        seq_len = data_element.shape[-2] + PRED_LEN # 75
        
        # Get hyperparameters if specified for training run
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["EncoderTransformerForClassification"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        epochs = hyperparam_vals.get("epochs", epochs)

        # Parameters for context transformer
        timeseries_element = data_train['data'][0][0][0][-1]
        #timeseries_context_element = data_train['data'][0][0][0][2]
        encoder_input_size = timeseries_element.shape[-1]
        context_len = 15
        seq_len = 15
        config_for_context_timeseries = get_config_for_context_timeseries(
            encoder_input_size, context_len, hyperparams)
        
        encoder_input_size = 512-1
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            encoder_input_size, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.num_labels = config_for_huggingface.num_labels

        model = GraphEncoderTransformerForClassification(
            config_for_huggingface, config_for_timeseries_lib,
            config_for_context_timeseries=config_for_context_timeseries,
            dataset_name=kwargs["model_opts"]["dataset_full"],
            model_opts=kwargs["model_opts"]
        )
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
            get_image_transform=True, img_model_config=None,
            #get_seg_maps_transforms=True,
            dataset_statistics=dataset_statistics)

        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            #evaluation_strategy="epoch",
            #save_strategy="epoch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            learning_rate=lr,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1,
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn
        )

        # Train model
        if not test_only:
            print("Starting training of model GAM Branch (Graph Attention Module) ===========================")
            trainer.train()

        return {
            "trainer": trainer,
            "val_transform": val_transforms_dicts
        }

    def test(self,
             test_data: tuple,
             training_result: dict,
             model_info: dict,
             *args,
             dataset_name: str = "",
             generator: bool = False,
             test_only: bool = False,
             **kwargs):
        """ Test model
        Args:
            test_data [tuple]: tuple containing data (index 0) and targets (index 1)
            training_result [dict]: dictionary containing training results
            model_info [dict]: dict containing model info such as saved path and transforms
            dataset_name [str]: name of dataset
            generator [bool]: if set to true, input data is provided in a generator
        """
        
        print("Starting inference using trained model Trajectory Transformer Classifier ===========================")

        if test_only:
            pretrained_model = load_pretrained_gam_branch(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class GraphEncoderTransformerForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict = None,
                 dataset_name: str = None,
                 model_opts: dict = None,
                 config_for_context_timeseries = None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self.num_labels = config_for_huggingface.num_labels

        classifier_hidden_size = 40 # config_for_timeseries_lib.num_class # number of neurons in last linear layer at the end of model
        
        self.graph_tf = GraphEncoderTransformer(
            config_for_huggingface,
            config_for_timeseries_lib,
            config_for_context_timeseries=config_for_context_timeseries
        )

        self.fc1 = nn.Linear(classifier_hidden_size, self.num_labels) # [40, 2]
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        timeseries_context: Optional[torch.Tensor] = None,
        previous_timeseries_context: Optional[torch.Tensor] = None,
        # normalized_trajectory_values: torch.Tensor = None,
        labels: torch.Tensor = None,
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

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.graph_tf(
            timeseries_context=timeseries_context
        )

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)
    

class GraphEncoderTransformer(TimeSeriesTransformerPreTrainedModel):

    base_model_prefix = "graph_tf"
    
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict = None,
                 dataset_name: str = None,
                 model_opts: dict = None,
                 config_for_context_timeseries = None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels
        self.timeseries_config = config_for_timeseries_lib

        self.context_transformer = EncoderTransformer(
            config_for_huggingface, config_for_context_timeseries)

        classifier_hidden_size = 40 # config_for_timeseries_lib.num_class # number of neurons in last linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()

        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)
        

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        timeseries_context: Optional[torch.Tensor] = None,
        labels: torch.Tensor = None,
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

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Apply "Graph" Transformer to scene graph data
        ctx_tf_output_0 = self.context_transformer(timeseries_context[:,0,:,:])
        ctx_tf_output_1 = self.context_transformer(timeseries_context[:,1,:,:])
        ctx_tf_output_2 = self.context_transformer(timeseries_context[:,2,:,:])
        ctx_tf_output_3 = self.context_transformer(timeseries_context[:,3,:,:])
        ctx_tf_output_4 = self.context_transformer(timeseries_context[:,4,:,:])
        ctx_tf_output_5 = self.context_transformer(timeseries_context[:,5,:,:])
        ctx_tf_output_6 = self.context_transformer(timeseries_context[:,6,:,:])
        ctx_tf_output_7 = self.context_transformer(timeseries_context[:,7,:,:])
        ctx_tf_output_8 = self.context_transformer(timeseries_context[:,8,:,:])
        ctx_tf_output_9 = self.context_transformer(timeseries_context[:,9,:,:])
        ctx_tf_output_10 = self.context_transformer(timeseries_context[:,10,:,:])
        ctx_tf_output_11 = self.context_transformer(timeseries_context[:,11,:,:])
        ctx_tf_output_12 = self.context_transformer(timeseries_context[:,12,:,:])
        ctx_tf_output_13 = self.context_transformer(timeseries_context[:,13,:,:])
        ctx_tf_output_14 = self.context_transformer(timeseries_context[:,14,:,:])

        ctx_tf_output = torch.stack((ctx_tf_output_0,
                                     ctx_tf_output_1,
                                     ctx_tf_output_2,
                                     ctx_tf_output_3,
                                     ctx_tf_output_4,
                                     ctx_tf_output_5,
                                     ctx_tf_output_6,
                                     ctx_tf_output_7,
                                     ctx_tf_output_8,
                                     ctx_tf_output_9,
                                     ctx_tf_output_10,
                                     ctx_tf_output_11,
                                     ctx_tf_output_12,
                                     ctx_tf_output_13,
                                     ctx_tf_output_14), dim=1)

        outputs = self.tsl_transformer(
            x_enc=ctx_tf_output,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        return outputs


class EncoderTransformer(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib
        ):
        super().__init__(config_for_huggingface)
        self.tsl_transformer_0 = TokengtTransformer(config_for_timeseries_lib[1]) 
        self.cross_attn_dim = 1

        self.cross_attention_1 = CrossAttention(self.cross_attn_dim) # CrossAttention(20)
        self.cross_attention_2 = CrossAttention(self.cross_attn_dim)
        self.cross_attention_3 = CrossAttention(self.cross_attn_dim)
        self.cross_attention_4 = CrossAttention(self.cross_attn_dim)

        self.road_emb_fc = nn.Linear(9, 5) # (9, self.cross_attn_dim)
        self.sidewalk_emb_fc = nn.Linear(4, 5)
        self.pedestrians_emb_fc = nn.Linear(8, 5)
        self.vehicles_emb_fc = nn.Linear(8, 5)
        
        self.use_cross_attention = False
        self.cross_attention_with_context_as_last_dim = False
        self.cross_attention_with_context_as_second_dim = True
        self.residual_at_the_end = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        *args,
        **kwargs
    ):
        if self.use_cross_attention:
            trajectory_values = self._apply_cross_attention(trajectory_values)

        # Get vanilla transformer model output
        outputs = self.tsl_transformer_0(
            x_enc=trajectory_values,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        return outputs
    
    def _apply_cross_attention(self, trajectory_values):
        """Apply inter-modal cross-attention to trajectory values.
        """
        # Compute inter-modal cross-attention
        road_values = nn.ReLU()(self.road_emb_fc(
            torch.squeeze(trajectory_values[:,0:9,:], -1)))
        sidewalk_values = nn.ReLU()(self.sidewalk_emb_fc(
            torch.squeeze(trajectory_values[:,9:13,:], -1)))
        pedestrians_values = nn.ReLU()(self.pedestrians_emb_fc(
            torch.squeeze(trajectory_values[:,13:21,:], -1)))
        vehicles_values = nn.ReLU()(self.vehicles_emb_fc(
            torch.squeeze(trajectory_values[:,21:29,:], -1)))
        
        if self.cross_attention_with_context_as_second_dim:
            unsqueeze_dim = 2 # shape=(batch, dim, 1)
        if self.cross_attention_with_context_as_last_dim:
            unsqueeze_dim = 1 # shape=(batch, 1, dim)
        road_values = road_values.unsqueeze(unsqueeze_dim) 
        sidewalk_values = sidewalk_values.unsqueeze(unsqueeze_dim)
        pedestrians_values = pedestrians_values.unsqueeze(unsqueeze_dim)
        vehicles_values = vehicles_values.unsqueeze(unsqueeze_dim)
        
        cross_attn_ctx_1, attn_1 = self.cross_attention_1(road_values, pedestrians_values)
        cross_attn_ctx_2, attn_2 = self.cross_attention_2(road_values, vehicles_values)
        cross_attn_ctx_3, attn_3 = self.cross_attention_3(sidewalk_values, pedestrians_values)
        cross_attn_ctx_4, attn_4 = self.cross_attention_4(pedestrians_values, vehicles_values)
        
        if self.cross_attention_with_context_as_last_dim:
            cross_attn_ctx_1 = cross_attn_ctx_1.swapaxes(1,2)
            cross_attn_ctx_2 = cross_attn_ctx_2.swapaxes(1,2)
            cross_attn_ctx_3 = cross_attn_ctx_3.swapaxes(1,2)
            cross_attn_ctx_4 = cross_attn_ctx_4.swapaxes(1,2)

        trajectory_values = torch.cat([
            trajectory_values, cross_attn_ctx_1, cross_attn_ctx_2,
            cross_attn_ctx_3, cross_attn_ctx_4], dim=1)
        
        return trajectory_values

def load_pretrained_gam_branch(dataset_name: str,
                               add_classification_head: bool = True,
                               submodels_paths: dict = None,
                               submodels_paths_override: str = None):
    config_for_encoder_tf = get_config_for_timeseries_lib(
            encoder_input_size=512-1, seq_len=15, hyperparams={})
    
    config_for_context_timeseries = get_config_for_context_timeseries(
        encoder_input_size=512-1, seq_len=15, hyperparams={})

    if submodels_paths_override:
        checkpoint = submodels_paths_override["gam_branch_path"]
    elif submodels_paths:
        checkpoint = submodels_paths["gam_branch_path"]
    else:
        if dataset_name == "combined":
            checkpoint = "data/models/combined/GAMBranch/09Apr2025-16h37m56s_CO9"
        if dataset_name in "pie":
            checkpoint = "_data/models/pie/GAMBranch/weights_gambranch_pie"
        elif dataset_name == "jaad_all":
            checkpoint = "data/models/jaad_all/GAMBranch/weights_gambranch_jaadall"
        elif dataset_name == "jaad_beh":
            checkpoint = "data/models/jaad_beh/TrajectoryTransformerb/weights_trajectorytransformerb_jaadbeh"

    if add_classification_head:
        pretrained_model = GraphEncoderTransformerForClassification.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_encoder_tf,
            config_for_context_timeseries=config_for_context_timeseries,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name)
    else:
        pretrained_model = GraphEncoderTransformer.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_encoder_tf,
            config_for_context_timeseries=config_for_context_timeseries,
            ignore_mismatched_sizes=True,
            #dataset_name=dataset_name,
            #submodels_paths=submodels_paths)
        )
    
    # Make all layers untrainable
    for child in pretrained_model.children():
        for param in child.parameters():
            param.requires_grad = False
    return pretrained_model


def get_config_for_timeseries_lib(encoder_input_size: int, 
                                  seq_len: int,
                                  hyperparams: dict, 
                                  pred_len: int = None):
    
    encoder_input_size = encoder_input_size + 1
    if hyperparams:
        hyperparams = hyperparams["EncoderTransformerForClassification"]

    # time series lib (TSLib) properties
    time_series_dict = {
        "task_name": "classification",
        "pred_len": pred_len if pred_len else PRED_LEN,
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": encoder_input_size, # encoder input size - default value,
        "d_model": hyperparams.get("d_model", 128), # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": 0.1, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 12), # num of heads
        "d_ff": hyperparams.get("d_ff", 1024), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 6), # num of encoder layers (or 3)
        "seq_len": seq_len, # input sequence length
        "num_class": 40, # number of neurons in last Linear layer at the end of model
        "label_len": 15,
        "c_out": 5,
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    return config_for_timeseries_lib


def get_config_for_context_timeseries(encoder_input_size, seq_len, hyperparams):

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
        "n_heads": hyperparams.get("n_heads", 12), # num of heads
        "d_ff": hyperparams.get("d_ff", 1024), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 6), # num of encoder layers (or 3)
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