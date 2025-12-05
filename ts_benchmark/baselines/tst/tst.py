import math
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
import torch.nn as nn
from ts_benchmark.baselines.tst.utils.pos_encoding import get_pos_encoder


# Model hyperparameters
MODEL_HYPER_PARAMS = {
    "d_model": 128,
    "n_heads": 4,             
    "num_layers": 2,          
    "dim_feedforward": 256,   
    "dropout": 0.1,           
    "pos_encoding": "sinespe"
}

class SimplifiedTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.project_inp = nn.Linear(config.enc_in, config.d_model)

        self.pos_enc = get_pos_encoder(config.pos_encoding)(config.d_model,config.dropout,config.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.time_projector = nn.Linear(config.seq_len, config.pred_len)
        self.feature_projector = nn.Linear(config.d_model, config.c_out)

    def forward(self, x):
        x = self.project_inp(x)         # [batch, seq_len, d_model]
        x = self.pos_enc(x)             # Add PE
        x = self.transformer_encoder(x) # [batch, seq_len, d_model]
        x = x.permute(0, 2, 1)          
        x = self.time_projector(x)      
        x = x.permute(0, 2, 1)          
        x = self.feature_projector(x)   
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
        """
        Defines the forward pass for the model.
        """
        output = self.model(input)
        return {"output": output}
