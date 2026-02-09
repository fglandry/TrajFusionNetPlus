import copy
from typing import Optional

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from libs.time_series_library.models_tsl.Tokengt import Model as TokengtTransformer
from libs.time_series_library.models_tsl.TransformerV2 import Model as VanillaTransformerTSLModel
from models.custom_layers_pytorch import CrossAttention
from models.hugging_face.model_trainers.graphtransformer import load_pretrained_graph_transformer
from models.hugging_face.model_trainers.trajectorytransformer import load_pretrained_trajectory_transformer
from models.hugging_face.model_trainers.trajectorytransformerb import load_pretrained_encoder_transformer
from models.hugging_face.model_trainers.van import load_pretrained_van
from models.hugging_face.model_trainers.vansequential import load_pretrained_van_sequential
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from utils.data_load import DataGenerator

PRED_LEN = 60
NET_INNER_DIM = 512
DROPOUT = 0.1


class VANSequentialV2(HuggingFaceTimeSeriesModel):

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
        
        print("Starting model loading for model Trajectory Transformer Classifier ======================")

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
        context_len = 15 # timeseries_context_element.shape[-2]
        seq_len = 4 # 15
        #config_for_context_timeseries = get_config_for_context_timeseries(
        #    encoder_input_size, context_len, hyperparams)
        
        encoder_input_size = 512-1 # 1024-1 # 512-1
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            encoder_input_size, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.num_labels = config_for_huggingface.num_labels

        model = VANEncoderTransformerForClassification(
            config_for_huggingface, config_for_timeseries_lib,
            #config_for_context_timeseries=config_for_context_timeseries,
            dataset_name=kwargs["model_opts"]["dataset_full"],
            model_opts=kwargs["model_opts"]
        )
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
            get_image_transform=True, img_model_config=None,
            get_seg_maps_transforms=True,
            dataset_statistics=dataset_statistics)

        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            #evaluation_strategy="steps",
            #save_strategy="steps",
            #eval_steps=100,
            #save_steps=100,
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
            print("Starting training of model Trajectory Transformer Classifier ===========================")
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
            pretrained_model = load_pretrained_van_sequential_v2(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class VANEncoderTransformerForClassification(TimeSeriesTransformerPreTrainedModel):
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
        
        self.van_sequential_tf = VANEncoderTransformer(
            config_for_huggingface,
            config_for_timeseries_lib,
            config_for_context_timeseries=config_for_context_timeseries,
            dataset_name=dataset_name,
            model_opts=model_opts
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
        video_context: Optional[torch.Tensor] = None,
        video_segmentation: Optional[torch.Tensor] = None,
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

        outputs = self.van_sequential_tf(
            timeseries_context=timeseries_context,
            video_context=video_context,
            video_segmentation=video_segmentation
        )

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)
    

class VANEncoderTransformer(TimeSeriesTransformerPreTrainedModel):

    base_model_prefix = "van_sequential_tf"
    
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

        
        # Get pretrained VAN Models -------------------------------------------
        """
        self.van = load_pretrained_van(
            dataset_name,
            is_predicted_overlays=False,
            add_classification_head=False,
            #submodels_paths=submodels_paths
        )
        """
        if dataset_name == "jaad_all":
            #van_min15_path = "data/models/jaad_all/VAN/21Apr2025-12h07m13s_VAN9"
            #van_min10_path = "data/models/jaad_all/VAN/19May2025-13h54m53s_VAN10B" # "data/models/jaad_all/VAN/21Apr2025-17h52m06s_VAN10"
            #van_min5_path = "data/models/jaad_all/VAN/19May2025-12h26m39s_VAN11B" # "data/models/jaad_all/VAN/23Apr2025-16h56m56s_VAN11"
            #van_0_path = "data/models/jaad_all/VAN/weights_van1_jaadall"
            van_min15_path = "data/models/jaad_all/VAN/weights_van_min15_jaad"
            van_min10_path = "data/models/jaad_all/VAN/weights_van_min10_jaad"
            van_min5_path = "data/models/jaad_all/VAN/weights_van_min5_jaad"
            van_0_path = "data/models/jaad_all/VAN/weights_van_0_jaad"
        elif dataset_name == "pie":
            #van_min15_path = "data/models/pie/VAN/23May2025-22h52m30s_VA14"
            #van_min10_path = "data/models/pie/VAN/23May2025-23h28m50s_VA15"
            #van_min5_path = "data/models/pie/VAN/24May2025-00h23m03s_VA16"
            #van_0_path = "data/models/pie/VAN/24May2025-10h21m43s_VA17"
            van_min15_path = "data/models/pie/VAN/weights_van_min15_pie"
            van_min10_path = "data/models/pie/VAN/weights_van_min10_pie"
            van_min5_path = "data/models/pie/VAN/weights_van_min5_pie"
            van_0_path = "data/models/pie/VAN/weights_van_0_pie"
        elif dataset_name == "combined":
            van_min15_path = "data/models/combined/VAN/06Jun2025-21h23m59s_CO7a"
            van_min10_path = "data/models/combined/VAN/07Jun2025-11h10m06s_CO7b"
            van_min5_path = "data/models/combined/VAN/07Jun2025-17h07m55s_CO7c"
            van_0_path = "data/models/combined/VAN/05Apr2025-09h52m52s_CO7"

        self.van_min15 = load_pretrained_van(dataset_name, is_predicted_overlays=True,
            add_classification_head=False,
            submodels_paths={"van_path": van_min15_path}
        )
        self.van_min10 = load_pretrained_van(dataset_name, is_predicted_overlays=True,
            add_classification_head=False,
            submodels_paths={"van_path": van_min10_path}
        )
        self.van_min5 = load_pretrained_van(dataset_name, is_predicted_overlays=True,
            add_classification_head=False,
            submodels_paths={"van_path": van_min5_path}
        )
        self.van_0 = load_pretrained_van(dataset_name, is_predicted_overlays=True,
            add_classification_head=False,
            submodels_paths={"van_path": van_0_path}
        )

        #self.van_channels = load_pretrained_van_sequential(dataset_name,
        #    add_classification_head=False)
        #self.van_channels_emb = nn.Linear(512, 40)


        # Get pretrained GraphTransformer model -------------------------------------------
        #self.graph_tf = load_pretrained_graph_transformer(
        #    dataset_name,
        #    add_classification_head=False,
        #    #submodels_paths=submodels_paths
        #)

        classifier_hidden_size = 40 # config_for_timeseries_lib.num_class # number of neurons in last linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()
        #self.classifier = nn.Linear(4096, classifier_hidden_size)

        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)
        

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        timeseries_context: Optional[torch.Tensor] = None,
        previous_timeseries_context: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        video_context_contains_full_sequence = True,
        video_segmentation: Optional[torch.Tensor] = None,
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

        if video_context_contains_full_sequence:
            van_output_0 = self.van_min15(video_context[:,-15,:,:,:]).pooler_output
            van_output_5 = self.van_min10(video_context[:,-10,:,:,:]).pooler_output
            van_output_10 = self.van_min5(video_context[:,-5,:,:,:]).pooler_output
            van_output_14 = self.van_0(video_context[:,-1,:,:,:]).pooler_output
        else:
            van_output_0 = self.van_min15(video_context[:,-4,:,:,:]).pooler_output
            van_output_5 = self.van_min10(video_context[:,-3,:,:,:]).pooler_output
            van_output_10 = self.van_min5(video_context[:,-2,:,:,:]).pooler_output
            van_output_14 = self.van_0(video_context[:,-1,:,:,:]).pooler_output

        ctx_tf_output = torch.stack((van_output_0,
                                     van_output_5,
                                     van_output_10,
                                     van_output_14), dim=1)

        outputs = self.tsl_transformer(
            x_enc=ctx_tf_output,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        
        """
        van_output_channels = self.van_channels(
            video_context=video_context
        )
        van_output_channels = self.van_channels_emb(van_output_channels)
        outputs = torch.cat((outputs, van_output_channels), dim=1)
        
        #x = ctx_tf_output.flatten(1)
        #outputs = self.classifier(x)
        """


        return outputs


def load_pretrained_van_sequential_v2(dataset_name: str,
                                        add_classification_head: bool = True,
                                        submodels_paths: dict = None):
    #config_for_encoder_tf = get_config_for_timeseries_lib(
    #        encoder_input_size=512-1, seq_len=15, hyperparams={})
    config_for_encoder_tf = get_config_for_timeseries_lib(
            encoder_input_size=512-1, seq_len=4, hyperparams={})
    
    #config_for_context_timeseries = get_config_for_context_timeseries(
    #    encoder_input_size=512-1, seq_len=15, hyperparams={})

    if submodels_paths:
        checkpoint = submodels_paths["enc_tf_path"]
    else:
        if dataset_name == "combined":
            checkpoint = "data/models/combined/VANSequentialV2/13Jun2025-20h24m35s_C15/checkpoint-10850"
        if dataset_name in "pie":
            checkpoint = "data/models/pie/VANSequentialV2/24May2025-11h06m42s_VAS4"
            checkpoint = "data/models/pie/VANSequentialV2/weights_vansequential_pie"
        elif dataset_name == "jaad_all":
            checkpoint = "data/models/jaad_all/VANSequentialV2/25Apr2025-09h43m46s_VAS1"
            checkpoint = "data/models/jaad_all/VANSequentialV2/19May2025-18h35m55s/checkpoint-539"
            checkpoint = "data/models/jaad_all/VANSequentialV2/19May2025-18h35m55s/checkpoint-4851"
            checkpoint = "data/models/jaad_all/VANSequentialV2/21May2025-08h54m54s/checkpoint-5390"
            checkpoint = "data/models/jaad_all/VANSequentialV2/19May2025-21h18m46s_VAS2"
            checkpoint = "data/models/jaad_all/VANSequentialV2/weights_vansequential_jaad"
        elif dataset_name == "jaad_beh":
            checkpoint = "data/models/jaad_beh/TrajectoryTransformerb/weights_trajectorytransformerb_jaadbeh"

    if add_classification_head:
        pretrained_model = VANEncoderTransformerForClassification.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_encoder_tf,
            #config_for_context_timeseries=config_for_context_timeseries,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name)
    else:
        pretrained_model = VANEncoderTransformer.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_encoder_tf,
            #config_for_context_timeseries=config_for_context_timeseries,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name,
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
    
    encoder_input_size = encoder_input_size + 1 # plus one to account for type identifiers
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
