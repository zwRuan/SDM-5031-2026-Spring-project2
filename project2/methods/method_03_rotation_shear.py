"""Method 3: Rotation + Shear Transformation
Combines Halton-sequence rotations with small deterministic shear for richer diversity.
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

    def halton(i, base=2):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    extras = []
    for i in range(aug_factor - 8):
        angle = halton(i + 1, 2) * 2 * np.pi
        shear = 0.1 * np.sin(halton(i + 1, 3) * 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        sheared_x = centered[:, :, 0] + shear * centered[:, :, 1]
        sheared_y = centered[:, :, 1]
        xr = sheared_x * c - sheared_y * s
        yr = sheared_x * s + sheared_y * c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx - mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - mn) / sc)
    return torch.cat(dihedral + extras, dim=0)
