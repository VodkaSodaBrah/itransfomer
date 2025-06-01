import pytest
import torch
from iTransformer.layers.Embed import TimeFeatureEmbedding, DataEmbedding_inverted

class TestTimeFeatureEmbedding:
    def test_time_feature_embedding(self):
        """Test TimeFeatureEmbedding with float features"""
        d_model = 512
        freq = '1min'
        
        # Create embedding
        time_embedding = TimeFeatureEmbedding(d_model, freq=freq)
        
        # Create input tensor (float features rather than indices)
        batch_size = 4
        seq_len = 24
        features = 4  # '1min' frequency has 4 features
        
        x_mark = torch.rand(batch_size, seq_len, features)
        
        # Forward pass
        output = time_embedding(x_mark)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_multiple_frequency_support(self):
        """Test that different frequencies are supported"""
        d_model = 64
        
        # Test various frequencies
        frequencies = ['1min', 'h', 'd', 'w', 'm']
        
        for freq in frequencies:
            # Create embedding
            time_embedding = TimeFeatureEmbedding(d_model, freq=freq)
            
            # Get expected feature count from the freq_map
            # This is implementation-dependent, so we're being careful here
            try:
                freq_map = getattr(time_embedding, 'freq_map', {'1min': 4, 'h': 4, 'd': 3, 'w': 2, 'm': 1})
                d_inp = freq_map.get(freq, 4)  # Default to 4 if not found
            except:
                # If we can't access freq_map, use reasonable defaults
                d_inp = 4 if freq in ['1min', 'h'] else 3 if freq == 'd' else 2 if freq == 'w' else 1
            
            # Create input tensor with appropriate feature dimension
            batch_size = 2
            seq_len = 12
            
            x_mark = torch.rand(batch_size, seq_len, d_inp)
            
            # Forward pass
            output = time_embedding(x_mark)
            
            # Check output shape
            assert output.shape == (batch_size, seq_len, d_model), f"Failed for frequency '{freq}'"


class TestDataEmbeddingInverted:
    def test_data_embedding_inverted_with_dropout(self):
        """Test DataEmbedding_inverted with dropout enabled"""
        # In iTransformer, c_in is the sequence length
        seq_len = 5    # This is what c_in actually represents
        d_model = 64   # Embedding dimension
        dropout = 0.1  # Dropout rate
        
        # Create embedding layer
        embedding = DataEmbedding_inverted(
            c_in=seq_len,   # Pass sequence length as c_in
            d_model=d_model,
            dropout=dropout,
            freq='1min',
            is_mps_compiled_with_dropout_disabled=False
        )
        
        # Create input tensors
        batch_size = 8
        feature_dim = 24   # Number of variables/features
        
        # Input tensor [batch, seq_len, features]
        x = torch.rand(batch_size, seq_len, feature_dim)
        
        # Marker tensor for timestamps [batch, seq_len, time_features]
        x_mark = torch.rand(batch_size, seq_len, 4)  # 4 time features for '1min'
        
        # Forward pass
        output = embedding(x, x_mark)
        
        # Check output shape
        # The output should be [batch, feature_dim, d_model]
        assert output.dim() == 3
        assert output.size(0) == batch_size
        assert output.size(1) == feature_dim, f"Expected feature dim {feature_dim}, got {output.size(1)}"
        assert output.size(2) == d_model, f"Expected model dim {d_model}, got {output.size(2)}"
    
    def test_dropout_disabled_for_mps(self):
        """Test that dropout is disabled when is_mps_compiled_with_dropout_disabled=True"""
        # In iTransformer, c_in is the sequence length
        seq_len = 5    # This is what c_in actually represents
        d_model = 64   # Embedding dimension
        dropout = 0.1  # Dropout rate
        
        # Create embedding with dropout disabled
        embedding = DataEmbedding_inverted(
            c_in=seq_len,   # Pass sequence length as c_in
            d_model=d_model,
            dropout=dropout,
            freq='1min',
            is_mps_compiled_with_dropout_disabled=True
        )
        
        # Verify the dropout layer is nn.Identity
        from torch.nn import Identity
        assert isinstance(embedding.dropout, Identity), "Dropout should be Identity when disabled for MPS"

        # Create input tensors
        batch_size = 4
        feature_dim = 12   # Number of variables/features
        
        # Input tensor [batch, seq_len, features]
        x = torch.rand(batch_size, seq_len, feature_dim)
        
        # Marker tensor for timestamps
        x_mark = torch.rand(batch_size, seq_len, 4)
        
        # Forward pass
        output = embedding(x, x_mark)
        
        # Basic shape check
        assert output.dim() == 3
        assert output.size(0) == batch_size
        assert output.size(1) == feature_dim
        assert output.size(2) == d_model