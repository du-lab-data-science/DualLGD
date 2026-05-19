import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    mask = mask.bool()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)

    dim = kwargs.get('dim', -1)
    x_masked = x.masked_fill(~mask, torch.finfo(x.dtype).min)
    probs = torch.softmax(x_masked, **kwargs)
    probs = probs.masked_fill(~mask, 0.0)

    valid_rows = mask.any(dim=dim, keepdim=True)
    probs = probs.masked_fill(~valid_rows, 0.0)
    return probs