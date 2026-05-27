"""Method 6: VDC-Dihedral Expansion
Van der Corput angles with transpose expansion (rotation + x-y swap).
"""
import torch
import numpy as np


def generate_augmentations(coords, aug_factor=256):
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    dihedral = [
        torch.cat((x, y), dim=2), torch.cat((1-x, y), dim=2),
        torch.cat((x, 1-y), dim=2), torch.cat((1-x, 1-y), dim=2),
        torch.cat((y, x), dim=2), torch.cat((1-y, x), dim=2),
        torch.cat((y, 1-x), dim=2), torch.cat((1-y, 1-x), dim=2),
    ]
    if aug_factor <= 8:
        return torch.cat(dihedral[:aug_factor], dim=0)

    def vdc(i, base=2):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    extras = []
    num_remaining = aug_factor - 8
    num_angles = (num_remaining + 1) // 2
    for i in range(num_angles):
        angle = vdc(i + 1) * 2 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:, :, 0] * c - centered[:, :, 1] * s
        yr = centered[:, :, 0] * s + centered[:, :, 1] * c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx - mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        nr = (rotated - mn) / sc
        extras.append(nr)
        if len(extras) >= num_remaining:
            break
        extras.append(torch.stack([nr[:, :, 1], nr[:, :, 0]], dim=2))
        if len(extras) >= num_remaining:
            break
    return torch.cat(dihedral + extras[:num_remaining], dim=0)
