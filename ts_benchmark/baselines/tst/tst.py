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

        pos_encoder_class = get_pos_encoder(config.pos_encoding)
        pos_encoder_args = {
            "d_model": config.d_model,
            "dropout": config.dropout,
            "max_len": config.seq_len,
        }
        # If using SineSPE and a period is provided in the config, add it to the arguments.
        if config.pos_encoding == 'sinespe' and hasattr(config, 'period') and config.period > 1:
            pos_encoder_args['period'] = config.period
        self.pos_enc = pos_encoder_class(**pos_encoder_args)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.d_model, config.c_out)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.project_inp(x)  # Project features to d_model: [batch, seq_len, d_model]
        x = self.pos_enc(x)      # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through the encoder: [batch, seq_len, d_model]
        x = x[:, -self.config.pred_len:, :]  # Take last pred_len steps for forecasting
        x = self.head(x) 
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
