"""Method 8: Prime-Sqrt Multi-Base Rotations
Cycles through sqrt of prime numbers as irrational multipliers for angle generation.
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

    primes_sqrt = [np.sqrt(p) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]]
    extras = []
    for i in range(aug_factor - 8):
        base = primes_sqrt[i % len(primes_sqrt)]
        angle = ((i + 1) * base % 1.0) * 2 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        centered = coords - 0.5
        xr = centered[:, :, 0] * c - centered[:, :, 1] * s
        yr = centered[:, :, 0] * s + centered[:, :, 1] * c
        rotated = torch.stack([xr, yr], dim=2)
        mn = rotated.min(dim=1, keepdim=True).values
        mx = rotated.max(dim=1, keepdim=True).values
        sc = (mx - mn).max(dim=2, keepdim=True).values.clamp(min=1e-8)
        extras.append((rotated - mn) / sc)
    return torch.cat(dihedral + extras, dim=0)
