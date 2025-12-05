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
    "pos_encoding": "learned",  # 'fixed' or 'learned
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

        self.project_inp = nn.Linear(config.enc_in, config.d_model)
        self.pos_enc = get_pos_encoder(config.pos_encoding)(config.d_model, config.dropout, config.seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.seq_len, config.pred_len)
        self.output_projection = nn.Linear(config.d_model, config.c_out)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.project_inp(x)  # Project features to d_model: [batch, seq_len, d_model]
        x = self.pos_enc(x)      # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through the encoder: [batch, seq_len, d_model]
        x = x.permute(0, 2, 1)    # [batch, d_model, seq_len]
        x = self.head(x)          # [batch, d_model, pred_len]
        x = x.permute(0, 2, 1)    # [batch, pred_len, d_model]
        x = self.output_projection(x) # [batch, pred_len, c_out]
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

