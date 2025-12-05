from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
import torch.nn as nn
from ts_benchmark.baselines.tst.utils.pos_encoding import get_pos_encoder


# Model hyperparameters
MODEL_HYPER_PARAMS = {
    # `max_len` is set automatically to self.config.seq_len
    "d_model": 128,
    "n_heads": 8,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "pos_encoding": "sinespe",  # 'fixed' or 'learned
    "norm": "BatchNorm",

    # --- Head Type ---
    "head_type": "flatten",

    # --- Parameters for the framework ---
    # `seq_len` will be used as `max_len` for the model
    "pred_len": 96,
}

class SimplifiedTST(nn.Module):
    """
    A simplified, self-contained, and robust Transformer model for forecasting.
    This version uses standard PyTorch components and is much more reliable.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Input Projection: Projects input features to the model dimension.
        self.project_inp = nn.Linear(config.enc_in, config.d_model)

        # 2. Positional Encoding: Adds time-step information.
        self.pos_enc = get_pos_encoder(config.pos_encoding)(config.d_model, config.dropout, config.seq_len)

        # 3. Standard PyTorch Transformer Encoder (uses LayerNorm by default)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True  # This simplifies the data flow, avoiding manual permutations.
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # 4. Forecasting Head: Maps the encoder output to the desired prediction length.
        self.head = nn.Linear(config.d_model * config.seq_len, config.pred_len * config.c_out)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.project_inp(x)  # Project features to d_model: [batch, seq_len, d_model]
        x = self.pos_enc(x)      # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through the encoder: [batch, seq_len, d_model]
        x = x.reshape(x.shape[0], -1)    # Flatten the output: [batch, seq_len * d_model]
        x = self.head(x)                 # Project to the forecast: [batch, pred_len * c_out]
        x = x.reshape(x.shape[0], self.config.pred_len, self.config.c_out) # Reshape to final output
        return x


class TST(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(TST, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    def _init_model(self):
        """
        Initializes the TST model with a forecasting head.
        """
        return SimplifiedTST(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)
        return {"output": output}

