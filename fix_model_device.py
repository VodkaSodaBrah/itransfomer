import os
import re

def patch_model_file():
    """Patch the iTransformer.py file to fix device handling"""
    model_path = "iTransformer/model/iTransformer.py"
    
    with open(model_path, 'r') as f:
        content = f.read()
    
    # Fix _detect_device method
    detect_device_pattern = r'def _detect_device\(self\):\s+""".*?"""\s+if torch\.backends\.mps\.is_available\(\):\s+.*?\s+elif torch\.cuda\.is_available\(\):\s+.*?\s+else:\s+return torch\.device\("cpu"\)'
    detect_device_replacement = '''def _detect_device(self):
        """Detect the best available device if not specified in configs"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")'''
    
    # Fix model initialization to handle device properly
    init_pattern = r'def __init__\(self, configs\):\s+super\(Model, self\).__init__\(\)\s+.*?self\.device = configs\.device if hasattr\(configs, \'device\'\) else self\._detect_device\(\)\s+.*?# Move model to the appropriate device\s+self\.to\(self\.device\)'
    init_replacement = '''def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Handle device in a safer way
        if hasattr(configs, 'device'):
            self.device = configs.device
        else:
            self.device = self._detect_device()
            
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
        # Don't move model in init - will be done in main script'''
    
    # Use regex with DOTALL flag to match across multiple lines
    content = re.sub(detect_device_pattern, detect_device_replacement, content, flags=re.DOTALL)
    content = re.sub(init_pattern, init_replacement, content, flags=re.DOTALL)
    
    # Create backup
    backup_path = model_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            with open(model_path, 'r') as orig:
                f.write(orig.read())
        print(f"Created backup at {backup_path}")
    
    # Write patched file
    with open(model_path, 'w') as f:
        f.write(content)
    
    print(f"Patched {model_path} to fix device handling")

if __name__ == "__main__":
    patch_model_file()