def test_temporal_embedding_shape():
    import torch
    from iTransformer.layers.Embed import TemporalEmbedding

    batch_size = 1024
    seq_length = 5
    d_model = 512

    # Create a dummy input tensor with the shape (batch_size, seq_length, d_model)
    x_mark = torch.randn(batch_size, seq_length, d_model)
    
    # Initialize the TemporalEmbedding layer
    temporal_embedding = TemporalEmbedding(d_model)

    # Forward pass
    output = temporal_embedding(x_mark)

    # Check the output shape
    assert output.shape == (batch_size, d_model), f"Expected output shape {(batch_size, d_model)}, but got {output.shape}"