from typing import Any, Optional

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from models.hugging_face.model_trainers.trajectorytransformer import load_pretrained_trajectory_transformer
from models.hugging_face.model_trainers.trajectorytransformeronlyspeed import load_pretrained_trajectory_tf_only_speed
from models.hugging_face.model_trainers.trajectorytransformernospeed import load_pretrained_trajectory_transformer as load_pretrained_trajectory_tf_box
from models.hugging_face.model_trainers.van import load_pretrained_van
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from models.hugging_face.utils.create_optimizer import get_optimizer
from utils.data_load import DataGenerator

PRED_LEN = 60


class TrajectoryTransformerbWithVan(HuggingFaceTimeSeriesModel):

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
        encoder_input_size = 10 # data_element.shape[-1] + 40
        seq_len = 75 # data_element.shape[-2] + PRED_LEN # 75
        
        # Get hyperparameters if specified for training run
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["EncoderTransformerForClassification"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        epochs = hyperparam_vals.get("epochs", epochs)
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            encoder_input_size, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.num_labels = config_for_huggingface.num_labels

        model = EncoderTransformerForClassification(
            config_for_huggingface, config_for_timeseries_lib,
            dataset_name=kwargs["model_opts"]["dataset_full"],
            model_opts=kwargs["model_opts"]
        )
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
            get_image_transform=True, img_model_config=None,
            dataset_statistics=dataset_statistics)

        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
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

        """
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
        """
        if test_only:
            optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts)
            trainer = self._get_trainer(model, args, train_dataset, 
                                        val_dataset, optimizer, lr_scheduler)
        else:
            # Train model
            #print("Starting training of model TrajFusionNet ===========================")
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
            args: TrainingArguments, train_dataset,
            val_dataset, data_train: dict, train_opts: dict):
        
        best_metric = 0
        best_trainer = None
        half_epochs = round(epochs / 2)
        
        """
        # Run first part of training procedure with the VAM branch disabled for 15 epochs
        # to improve learning in the SAM branch.
        # In order to do this, the weights in the VAM projection layer ('van_output_embed')
        # as well as the associated learning rate are set to zero
        with torch.no_grad(): 
            model.base_model.transformer.van_output_embed.weight.zero_()
            model.base_model.transformer.van_output_embed.bias.zero_()

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
        """

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
            pretrained_model = load_pretrained_encoder_transformer(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class EncoderTransformerForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict = None,
                 dataset_name: str = None,
                 model_opts: dict = None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels
        self.timeseries_config = config_for_timeseries_lib

        self.transformer = EncoderTransformer(config_for_huggingface, config_for_timeseries_lib,
                                              dataset_name=dataset_name,
                                              model_opts=model_opts)

        classifier_hidden_size = config_for_timeseries_lib.num_class # number of neurons in last linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()

        self.fc1 = nn.Linear(classifier_hidden_size, self.num_labels) # [40, 2]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        normalized_trajectory_values: torch.Tensor = None,
        video_context: Optional[torch.Tensor] = None,
        labels: torch.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
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

        # Get encoder transformer model output
        outputs = self.transformer(
            trajectory_values,
            normalized_trajectory_values,
            video_context
        )

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)


class EncoderTransformer(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict,
                 dataset_name: str = None,
                 submodels_paths: dict = None,
                 model_opts: dict = None
        ):
        super().__init__(config_for_huggingface)
        model_opts = model_opts if model_opts else {}
        self._dataset = dataset_name
        self._device = get_device()

        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)


        self.traj_TF = load_pretrained_trajectory_transformer(dataset_name,
                                                              submodels_paths=submodels_paths,
                                                              submodels_paths_override=model_opts.get("submodels_paths_override"))

        self.van = load_pretrained_van(
            dataset_name,
            is_predicted_overlays=False,
            add_classification_head=False,
            #train_layers=True
        )

        #self.van_enc = nn.Linear(3 * 224 * 224, 40)
        self.van_output_embed = nn.Linear(512, 5)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values: torch.Tensor,
        normalized_trajectory_values: torch.Tensor,
        video_context: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """ Args:
        trajectory_values [torch.Tensor]: non-normalized observed trajectory values
            of shape [batch, seq_len, enc]
        normalized_trajectory_values [torch.Tensor]: normalized observed trajectory values
            of shape [batch, seq_len, enc]
        """

        # Trajectory prediction ====================================================
        
        # Predict trajectory (pedestrian bounding boxes and vehicle speed) between 
        # t=0 and t=60
        predicted_trajectory = self.traj_TF(
             normalized_trajectory_values=normalized_trajectory_values
        ).logits # [b, 60, 5]


        # Crossing prediction ====================================================

        # Add type identifier to indicate past observed trajectory (add 1) vs 
        # predicted trajectory (add 0)
        extra_dim_vals = torch.zeros((predicted_trajectory.shape[0],          # add 0 token
                                      predicted_trajectory.shape[1], 1)).to(self._device)
        predicted_trajectory = torch.cat((predicted_trajectory, extra_dim_vals), 2)
        extra_dim_vals = torch.ones((normalized_trajectory_values.shape[0],   # add 1 token
                                     normalized_trajectory_values.shape[1], 1)).to(self._device)
        trajectory_values = torch.cat((normalized_trajectory_values, extra_dim_vals), 2)

        predicted_trajectory = torch.cat([trajectory_values,
                                          predicted_trajectory], dim=1) # [b, 75, 6]

        # Apply VAN to each sequence index
        """
        outputs = []
        for t in range(video_context.shape[1]):  
            frame_t = video_context[:, t, :]  
            out_t = self.van(frame_t).pooler_output      
            outputs.append(out_t)
        sequence_output = torch.stack(outputs, dim=1)
        """
        van_output_0 = self.van(video_context[:, 0, :, :, :]).pooler_output
        #van_output_1 = self.van(video_context[:, 1, :, :, :]).pooler_output
        van_output_2 = self.van(video_context[:, 2, :, :, :]).pooler_output
        #van_output_3 = self.van(video_context[:, 3, :, :, :]).pooler_output
        van_output_4 = self.van(video_context[:, 4, :, :, :]).pooler_output
        #van_output_5 = self.van(video_context[:, 5, :, :, :]).pooler_output
        van_output_6 = self.van(video_context[:, 6, :, :, :]).pooler_output
        #van_output_7 = self.van(video_context[:, 7, :, :, :]).pooler_output
        van_output_8 = self.van(video_context[:, 8, :, :, :]).pooler_output
        #van_output_9 = self.van(video_context[:, 9, :, :, :]).pooler_output
        van_output_10 = self.van(video_context[:, 10, :, :, :]).pooler_output
        #van_output_11 = self.van(video_context[:, 11, :, :, :]).pooler_output
        van_output_12 = self.van(video_context[:, 12, :, :, :]).pooler_output
        #van_output_13 = self.van(video_context[:, 13, :, :, :]).pooler_output
        van_output_14 = self.van(video_context[:, 14, :, :, :]).pooler_output

        sequence_output = torch.stack([
            van_output_0,
            van_output_2, van_output_2,
            van_output_4, van_output_4,
            van_output_6, van_output_6,
            van_output_8, van_output_8,
            van_output_10, van_output_10,
            van_output_12, van_output_12, 
            van_output_14, van_output_14
        ], dim=1)

        """
        v_flat = sequence_output.view(16, 15, -1)
        van_encodings = self.van_enc(v_flat)
        """
        van_encodings = self.van_output_embed(sequence_output)
        # , 75, 6+40]

        # Copy last time step into the predicted timesteps
        last_step = van_encodings[:, -1:, :]
        repeated = last_step.repeat(1, 60, 1)
        van_encodings = torch.cat([van_encodings, repeated], dim=1) # [b, 75, 40]
        
        #sequence_output = self.van_enc(sequence_output)
        predicted_trajectory = torch.cat([predicted_trajectory, van_encodings], dim=2) # [b, 75, 6+40]

        # Get vanilla transformer model output
        outputs = self.tsl_transformer(
            x_enc=predicted_trajectory,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        return outputs # [b, 40]


def load_pretrained_encoder_transformer(dataset_name: str,
                                        add_classification_head: bool = True,
                                        submodels_paths: dict = None):
    config_for_encoder_tf = get_config_for_timeseries_lib(
            encoder_input_size=8, seq_len=75, hyperparams={})
    if submodels_paths:
        checkpoint = submodels_paths["enc_tf_path"]
    else:
        if dataset_name in ["pie", "combined"]:
            #checkpoint = "data/models/pie/TrajectoryTransformerb/weights_trajectorytransformerb_pie"
            checkpoint = "data/models/pie/TrajectoryTransformerbWithVan/30Apr2025-20h20m36s/checkpoint-598"
            checkpoint = "data/models/pie/TrajectoryTransformerbWithVan/01May2025-21h49m57s/checkpoint-8970"
        elif dataset_name == "jaad_all":
            checkpoint = "data/models/jaad_all/TrajectoryTransformerb/weights_trajectorytransformerb_jaadall"
        elif dataset_name == "jaad_beh":
            checkpoint = "data/models/jaad_beh/TrajectoryTransformerb/weights_trajectorytransformerb_jaadbeh"

    if add_classification_head:
        pretrained_model = EncoderTransformerForClassification.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_encoder_tf,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name)
    else:
        pretrained_model = EncoderTransformer.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_encoder_tf,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name,
            submodels_paths=submodels_paths)
    
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
