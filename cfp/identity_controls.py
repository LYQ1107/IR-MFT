"""Parameter-free bounded multiple-prototype identity baseline."""
import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


class MultiPrototypeMemory(nn.Module):
    def forward(self, features, frames, valid, query_frame):
        source, _, lengths, available, _ = checked_history(features, frames, valid, query_frame)
        result = torch.zeros_like(source[:, 0])
        for row, length in enumerate(lengths.tolist()):
            prototypes, counts, latest = [], [], []
            for step in range(length):
                feature = source[row, step]
                distances = torch.stack([1 - value @ feature for value in prototypes]) if prototypes else None
                if not prototypes or (len(prototypes) < 3 and float(distances.min()) > .2):
                    prototypes.append(feature.clone())
                    counts.append(1)
                    latest.append(int(frames[row, step]))
                else:
                    index = int(distances.argmin())
                    prototypes[index] = F.normalize(counts[index] * prototypes[index] + feature, dim=0)
                    counts[index] += 1
                    latest[index] = int(frames[row, step])
            if prototypes:
                selected = max(range(len(prototypes)), key=lambda index: (counts[index], latest[index]))
                result[row] = prototypes[selected]
        return result, available
