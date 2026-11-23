import torch


class AttentionCore(torch.nn.Module):
    """Unmasked scaled dot-product attention; no dropout or projections."""

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (q.shape[-1] ** -0.5)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v)
