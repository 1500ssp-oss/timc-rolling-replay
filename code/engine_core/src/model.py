
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Lightweight multi-head self-attention with [batch, seq, d_model] I/O."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        return self.dense(out)


class PatentLSTM(nn.Module):
    """Three LSTM encoders, multi-head attention, and a decoded output head.

    The three encoders receive the same input feature tensor in this archived
    implementation; their summed representation is coupled over time by the
    attention layer.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        pred_steps: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_dim = input_dim
        self.encoder_tension = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.encoder_thickness = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.encoder_flatness = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_tension = nn.Linear(hidden_dim, pred_steps)
        self.fc_thickness = nn.Linear(hidden_dim, pred_steps)
        self.fc_flatness = nn.Linear(hidden_dim, pred_steps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out_t, _ = self.encoder_tension(x)
        out_h, _ = self.encoder_thickness(x)
        out_s, _ = self.encoder_flatness(x)

        # Sum the three encoded representations before temporal attention.
        combined = out_t + out_h + out_s
        attn_out = self.attention(combined, combined, combined)
        attn_out = self.dropout(attn_out)

        # Decode from the last attended update to the requested prediction steps.
        decoder_input = attn_out[:, -1:, :]
        decoder_out, _ = self.decoder(decoder_input)
        z = self.dropout(decoder_out.squeeze(1))

        pred_t = self.fc_tension(z)
        pred_h = self.fc_thickness(z)
        pred_s = self.fc_flatness(z)
        return torch.stack([pred_t, pred_h, pred_s], dim=2)
