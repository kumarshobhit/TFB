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
    "pos_encoding": "learned",  # 'fixed' or 'learned
    "activation": "gelu",
    "norm": "BatchNorm",

    # --- Head Type ---
    "head_type": "projection",  # 'projection' or 'slicing'

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
        if config.head_type == 'projection':
            self.head = nn.Linear(config.seq_len, config.pred_len)


    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.encoder(x)
        if self.config.head_type == 'projection':
            # Project along the time dimension for each feature independently
            x = x.permute(0, 2, 1)      # -> [batch, features, seq_len]
            x = self.head(x)           # -> [batch, features, pred_len]
            return x.permute(0, 2, 1)  # -> [batch, pred_len, features]
        else:  # 'slicing' head
            # Take the last `pred_len` time steps of the encoder's output as the forecast
            return x[:, -self.config.pred_len:, :]


class TST(DeepForecastingModelBase):
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