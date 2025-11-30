from ts_benchmark.baselines.tst.models.transformer import TSTransformerEncoder
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
import torch.nn as nn

# Model hyperparameters
MODEL_HYPER_PARAMS = {
    # --- Parameters for DeepForecastingModelBase ---
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "patience": 5,
    "loss": "mse",
    "lradj": "type3",
    "num_workers": 0,

     # --- Parameters for TSTransformerEncoder ---
    # `feat_dim` is set automatically to self.config.enc_in
    # `max_len` is set automatically to self.config.seq_len
    "d_model": 128,
    "n_heads": 8,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "pos_encoding": "sinespe",  # 'fixed' or 'learnable'
    "activation": "gelu",
    "norm": "BatchNorm",

    # --- Parameters for the framework ---
    # `seq_len` will be used as `max_len` for the model
    "seq_len": 96,
    # `horizon` is the prediction horizon.
    # It's used to create the forecasting head.
    "pred_len": 96,
}

class TSTForecastingModel(nn.Module):
    """
    Attributes:
    A wrapper for TSTransformerEncoder that adds a forecasting head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TSTransformerEncoder(
            feat_dim=config.enc_in,
            max_len=config.seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            pos_encoding=config.pos_encoding,
            activation=config.activation,
            norm=config.norm,
        )
        self.head = nn.Linear(config.seq_len * config.enc_in, config.pred_len * config.c_out)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.encoder(x)  # [batch, seq_len, features]
        x = x.reshape(x.shape[0], -1)  # [batch, seq_len * features]
        x = self.head(x)  # [batch, pred_len * features]
        x = x.reshape(x.shape[0], self.config.pred_len, self.config.c_out)  # [batch, pred_len, features]
        return x



class TST(DeepForecastingModelBase):
    """
    DUET Adapter Class

    Attributes:
        model_name (str): Model identifier name for distinguishing different models.
        _init_model: Method to initialize an instance of DUETModel.
        _process: Executes the model's forward pass and returns the output.
    """
    def __init__(self, **kwargs):
        super(TST, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TST"

    def _init_model(self):
        """
        Initializes the TST model with a forecasting head.
        """
        return TSTForecastingModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        """
        Defines the forward pass for the model.
        """
        output = self.model(input)
        return {"output": output}