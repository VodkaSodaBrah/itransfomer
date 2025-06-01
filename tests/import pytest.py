import pytest
import torch

# filepath: /Users/mchildress/Active Code/itransformer/iTransformer/layers/test_Embed.py
import torch.nn as nn
from iTransformer.layers.Embed import (
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    DataEmbedding,
    DataEmbedding_inverted
)

class TestPositionalEmbedding:
    def test_init(self):
        """Test initialization of PositionalEmbedding"""
        d_model = 512
        max_len = 1000
        
        # Test with default max_len
        pe1 = PositionalEmbedding(d_model)
        assert hasattr(pe1, 'pe')
        assert pe1.pe.shape == (1, 5000, d_model)
        
        # Test with custom max_len
        pe2 = PositionalEmbedding(d_model, max_len)
        assert hasattr(pe2, 'pe')
        assert pe2.pe.shape == (1, max_len, d_model)

class TestTokenEmbedding:
    def test_init(self):
        """Test initialization of TokenEmbedding"""
        c_in = 5
        d_model = 512
        
        te = TokenEmbedding(c_in, d_model)
        assert hasattr(te, 'tokenConv')
        
        # Check conv layer properties
        assert isinstance(te.tokenConv, nn.Conv1d)
        assert te.tokenConv.in_channels == c_in
        assert te.tokenConv.out_channels == d_model
        assert te.tokenConv.kernel_size == (3,)

class TestFixedEmbedding:
    def test_init(self):
        """Test initialization of FixedEmbedding"""
        c_in = 24  # e.g., hours in a day
        d_model = 512
        
        fe = FixedEmbedding(c_in, d_model)
        assert hasattr(fe, 'emb')
        assert isinstance(fe.emb, nn.Embedding)
        assert fe.emb.weight.shape == (c_in, d_model)
        assert not fe.emb.weight.requires_grad  # Weight should be non-trainable

class TestTemporalEmbedding:
    def test_init_hourly(self):
        """Test initialization of TemporalEmbedding with hourly frequency"""
        d_model = 512
        embed_type = 'fixed'
        freq = 'h'
        
        te = TemporalEmbedding(d_model, embed_type, freq)
        
        # Check that appropriate embeddings are created
        assert hasattr(te, 'hour_embed')
        assert hasattr(te, 'weekday_embed')
        assert hasattr(te, 'day_embed')
        assert hasattr(te, 'month_embed')
        assert not hasattr(te, 'minute_embed')  # Should not have minute for hourly freq
        
        # Check embedding types
        assert isinstance(te.hour_embed, FixedEmbedding)
        assert te.hour_embed.emb.weight.shape == (24, d_model)  # 24 hours
    
    def test_init_minutely(self):
        """Test initialization of TemporalEmbedding with minutely frequency"""
        d_model = 512
        embed_type = 'fixed'
        freq = 't'
        
        te = TemporalEmbedding(d_model, embed_type, freq)
        
        # Check that appropriate embeddings are created
        assert hasattr(te, 'hour_embed')
        assert hasattr(te, 'weekday_embed')
        assert hasattr(te, 'day_embed')
        assert hasattr(te, 'month_embed')
        assert hasattr(te, 'minute_embed')  # Should have minute for minutely freq
        
        # Check embedding types
        assert isinstance(te.minute_embed, FixedEmbedding)
        assert te.minute_embed.emb.weight.shape == (4, d_model)  # 4 minute buckets
    
    def test_init_learnable(self):
        """Test initialization of TemporalEmbedding with learnable embeddings"""
        d_model = 512
        embed_type = 'learned'  # Not 'fixed', so should use nn.Embedding
        freq = 'h'
        
        te = TemporalEmbedding(d_model, embed_type, freq)
        
        # Check embedding types
        assert isinstance(te.hour_embed, nn.Embedding)
        assert te.hour_embed.weight.shape == (24, d_model)
        assert te.hour_embed.weight.requires_grad  # Should be trainable

class TestTimeFeatureEmbedding:
    def test_init(self):
        """Test initialization of TimeFeatureEmbedding"""
        d_model = 512
        
        # Test with different frequencies
        for freq, expected_dim in [('h', 4), ('t', 5), ('s', 6), ('m', 1), 
                                  ('w', 2), ('d', 3), ('1min', 4)]:
            tfe = TimeFeatureEmbedding(d_model, freq=freq)
            assert hasattr(tfe, 'embed')
            assert isinstance(tfe.embed, nn.Linear)
            assert tfe.embed.in_features == expected_dim
            assert tfe.embed.out_features == d_model
            assert tfe.embed.bias is None  # bias=False

class TestDataEmbedding:
    def test_init(self):
        """Test initialization of DataEmbedding"""
        c_in = 5
        d_model = 512
        
        # Test with fixed embedding type
        de1 = DataEmbedding(c_in, d_model, embed_type='fixed')
        assert hasattr(de1, 'value_embedding')
        assert hasattr(de1, 'position_embedding')
        assert hasattr(de1, 'temporal_embedding')
        assert hasattr(de1, 'dropout')
        
        assert isinstance(de1.value_embedding, TokenEmbedding)
        assert isinstance(de1.position_embedding, PositionalEmbedding)
        assert isinstance(de1.temporal_embedding, TemporalEmbedding)
        assert isinstance(de1.dropout, nn.Dropout)
        
        # Test with timeF embedding type
        de2 = DataEmbedding(c_in, d_model, embed_type='timeF')
        assert isinstance(de2.temporal_embedding, TimeFeatureEmbedding)

class TestDataEmbeddingInverted:
    def test_init_default(self):
        """Test initialization of DataEmbedding_inverted with default parameters"""
        c_in = 5
        d_model = 512
        
        dei = DataEmbedding_inverted(c_in, d_model)
        
        assert hasattr(dei, 'value_embedding')
        assert hasattr(dei, 'position_embedding')
        assert hasattr(dei, 'temporal_embedding')
        assert hasattr(dei, 'dropout')
        
        assert isinstance(dei.value_embedding, nn.Linear)
        assert dei.value_embedding.in_features == c_in
        assert dei.value_embedding.out_features == d_model
        
        assert isinstance(dei.position_embedding, nn.Linear)
        assert dei.position_embedding.in_features == c_in
        assert dei.position_embedding.out_features == d_model
        
        assert isinstance(dei.dropout, nn.Dropout)
        assert dei.dropout.p == 0.1  # Default dropout
    
    def test_init_mps_optimized(self):
        """Test initialization with MPS optimization flags"""
        c_in = 5
        d_model = 512
        
        # Test with MPS optimization enabled
        dei = DataEmbedding_inverted(
            c_in, d_model, 
            disable_dropout_for_mps_compile=True,
            is_mps_compiled_with_dropout_disabled=True
        )
        
        assert isinstance(dei.dropout, nn.Identity)  # Should be Identity when disabled
        
        # Test with MPS optimization disabled
        dei2 = DataEmbedding_inverted(
            c_in, d_model, 
            disable_dropout_for_mps_compile=True,
            is_mps_compiled_with_dropout_disabled=False
        )
        
        assert isinstance(dei2.dropout, nn.Dropout)  # Should still be Dropout