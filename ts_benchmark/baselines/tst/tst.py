from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
import torch
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
    # 'fixed' or 'learned' or 'rotary' or 'sinespe'
    "channel_independence": False,
    "revin": True,
    # --- Parameters for the framework ---
    # `seq_len` will be used as `max_len` for the model
    "pred_len": 96,
}

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_statistics(self):
        self.mean = None
        self.stdev = None

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.mean = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*0)
        x = x * self.stdev
        x = x + self.mean
        return x

class SimplifiedTST(nn.Module):
    """
    A simplified, self-contained, and robust Transformer model for forecasting.
    This version uses standard PyTorch components and is much more reliable.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Check for Channel Independence
        self.channel_independence = getattr(config, 'channel_independence', False)
        
        # RevIN
        self.revin = getattr(config, 'revin', True)
        if self.revin:
            self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)
        
        # If CI is enabled, we project 1 feature -> d_model. Otherwise, we project enc_in -> d_model.
        enc_in = 1 if self.channel_independence else config.enc_in
        self.project_inp = nn.Linear(enc_in, config.d_model)

        pos_encoder_class = get_pos_encoder(config.pos_encoding)
        pos_encoder_args = {
            "d_model": config.d_model,
            "dropout": config.dropout,
            "max_len": config.seq_len,
        }
        # If using SineSPE and a period is provided in the config, add it to the arguments.
        if config.pos_encoding == 'sinespe' and hasattr(config, 'period') and config.period > 1:
            pos_encoder_args['period'] = config.period
        # If using Rotary and base_freq is provided in the config, add it to the arguments.
        if config.pos_encoding == 'rotary' and hasattr(config, 'base_freq'):
            pos_encoder_args['base_freq'] = config.base_freq
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
        
        # If CI is enabled, the head outputs 1 value (per channel). Otherwise, it outputs c_out.
        head_out = 1 if self.channel_independence else config.c_out
        self.head = nn.Linear(config.d_model, head_out)

    def forward(self, x):
        # x: [batch, seq_len, features]
        
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        if self.channel_independence:
            # Reshape to treat each channel as an independent sample
            # [Batch, Seq, Vars] -> [Batch, Vars, Seq] -> [Batch * Vars, Seq, 1]
            B, L, C = x.shape
            x = x.permute(0, 2, 1).reshape(B * C, L, 1)
            
        x = self.project_inp(x)  # Project features to d_model: [batch, seq_len, d_model]
        x = self.pos_enc(x)      # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through the encoder: [batch, seq_len, d_model]
        x = x[:, -self.config.pred_len:, :]  # Take last pred_len steps for forecasting
        x = self.head(x) 
        
        if self.channel_independence:
            # Reshape back to original format: [Batch * Vars, Pred, 1] -> [Batch, Pred, Vars]
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            
        if self.revin:
            x = self.revin_layer(x, 'denorm')
            
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
