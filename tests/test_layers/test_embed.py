import pytest
import torch
from iTransformer.layers.Embed import TemporalEmbedding

def test_temporal_embedding_shape():
    batch_size = 1024
    seq_length = 5
    d_model = 512
    temporal_embedding = TemporalEmbedding(d_model)

    # Create a dummy input tensor with the shape (batch_size, seq_length, features)
    input_tensor = torch.randn(batch_size, seq_length, 4)  # Assuming 4 features for the input

    # Forward pass through the temporal embedding
    output = temporal_embedding(input_tensor)

    # Check the output shape
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

def test_temporal_embedding_integration():
    batch_size = 1024
    seq_length = 5
    d_model = 512
    temporal_embedding = TemporalEmbedding(d_model)

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, seq_length, 4)

    # Forward pass through the temporal embedding
    output = temporal_embedding(input_tensor)

    # Ensure the output is a tensor
    assert isinstance(output, torch.Tensor), "Output should be a tensor"