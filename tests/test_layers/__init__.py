def test_temporal_embedding():
    import torch
    from iTransformer.layers.Embed import TemporalEmbedding

    batch_size = 2
    seq_length = 5
    d_model = 512
    x_mark = torch.randn(batch_size, seq_length, 4)  # Example input with shape [B, C_mark, d_model]

    temporal_embedding = TemporalEmbedding(d_model)
    output = temporal_embedding(x_mark)

    assert output.shape == (batch_size, d_model), f"Expected output shape {(batch_size, d_model)}, but got {output.shape}"

def test_forward_pass():
    import torch
    from iTransformer.model.iTransformer import iTransformer

    batch_size = 2
    seq_length = 5
    d_model = 512
    x_enc = torch.randn(batch_size, seq_length, 5)  # Example input
    x_mark_enc = torch.randn(batch_size, seq_length, 4)  # Example timestamp input
    x_dec = torch.randn(batch_size, seq_length, 5)  # Example decoder input
    x_mark_dec = torch.randn(batch_size, seq_length, 4)  # Example decoder timestamp input

    model = iTransformer(d_model=d_model)
    output, attns = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

def test_metrics_util():
    from iTransformer.utils.metrics import calculate_metric

    # Example predictions and targets
    predictions = [0.5, 0.6, 0.7]
    targets = [0.5, 0.6, 0.8]

    metric = calculate_metric(predictions, targets)
    assert isinstance(metric, float), "Metric should be a float value" 

