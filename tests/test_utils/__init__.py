def test_temporal_embedding_shape():
    import torch
    from iTransformer.layers.Embed import TemporalEmbedding

    batch_size = 1024
    seq_length = 5
    d_model = 512
    temporal_embedding = TemporalEmbedding(d_model)

    x_mark = torch.randn(batch_size, seq_length, 4)  # Example input
    output = temporal_embedding(x_mark)

    assert output.shape == (batch_size, d_model), f"Expected output shape {(batch_size, d_model)}, but got {output.shape}"

def test_forward_shape():
    import torch
    from iTransformer.model.iTransformer import iTransformer

    batch_size = 1024
    seq_length = 5
    d_model = 512
    model = iTransformer(d_model)

    x_enc = torch.randn(batch_size, seq_length, 5)  # Example input
    x_mark_enc = torch.randn(batch_size, seq_length, 4)  # Example timestamp input
    x_dec = torch.randn(batch_size, seq_length, 5)  # Example input
    x_mark_dec = torch.randn(batch_size, seq_length, 4)  # Example timestamp input

    output, attns = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"