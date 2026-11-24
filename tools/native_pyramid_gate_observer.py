"""Read-only per-level gate/confidence/correction summaries, no retained tensors."""
import math

import torch
from torch.nn import functional as F


class Moments:
    def __init__(self):
        self.count, self.total, self.squares = 0, 0., 0.
        self.minimum = self.maximum = None

    def add(self, value):
        value = value.detach().double().reshape(-1)
        if not value.numel(): return
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite native diagnostic values')
        self.count += value.numel()
        self.total += float(value.sum())
        self.squares += float(value.square().sum())
        lo, hi = float(value.min()), float(value.max())
        self.minimum = lo if self.minimum is None else min(self.minimum, lo)
        self.maximum = hi if self.maximum is None else max(self.maximum, hi)

    def summary(self):
        if not self.count: return dict(count=0, mean=None, std=None, min=None, max=None)
        mean = self.total/self.count
        return dict(count=self.count, mean=mean, std=math.sqrt(max(0., self.squares/self.count-mean*mean)),
                    min=self.minimum, max=self.maximum)


class NativeGateObserver:
    def __init__(self):
        self.rows, self.handles = {}, []

    def attach(self, name, module):
        if name in self.rows: raise ValueError('One observer entry per native instance')
        row = dict(frames=0, available_frames=0, missing_frames=0,
            mode=module.mode, quality_gate_enabled=module.use_quality,
            feature_protection_enabled=module.protect_rgb, levels=[])
        for _ in range(3):
            row['levels'].append(dict(gate=Moments(), gain=Moments(), covariance=Moments(),
                valid_cells=0, changed_cells=0, delta_squared_sum=0., base_squared_sum=0.,
                reliable_cells=0, reliable_changed_cells=0))
        self.rows[name] = row
        def hook(net, args, kwargs, output):
            with torch.no_grad():
                row['frames'] += 1
                repaired, state, information = output
                if information['missing_exact_RGB']:
                    row['missing_frames'] += 1
                    return
                row['available_frames'] += 1
                rgb = args[0] if args else kwargs['rgb']
                scores = args[2] if len(args) > 2 else kwargs['scores']
                full_valid = kwargs.get('valid')
                for index, (base, value, score, fields) in enumerate(zip(rgb, repaired, scores, information['levels'])):
                    item = row['levels'][index]
                    valid = (torch.ones_like(base[:, :1], dtype=torch.bool) if full_valid is None else
                             F.adaptive_avg_pool2d(full_valid.float(), base.shape[-2:]) >= .5)
                    item['gate'].add(fields['gate'])
                    item['gain'].add(fields['gain'].expand_as(valid)[valid])
                    if fields['covariance'] is not None:
                        item['covariance'].add(fields['covariance'].expand_as(valid)[valid])
                    changed = (value.detach().float() != base.detach().float()).any(1, keepdim=True)
                    reliable = valid & (score.detach() >= .6)
                    item['valid_cells'] += int(valid.sum())
                    item['changed_cells'] += int((changed & valid).sum())
                    item['reliable_cells'] += int(reliable.sum())
                    item['reliable_changed_cells'] += int((changed & reliable).sum())
                    delta2 = (value.detach().float()-base.detach().float()).square().mean(1, keepdim=True)
                    base2 = base.detach().float().square().mean(1, keepdim=True)
                    item['delta_squared_sum'] += float(delta2[valid].double().sum())
                    item['base_squared_sum'] += float(base2[valid].double().sum())
        self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def summary(self, name):
        row = self.rows[name]
        levels = []
        for index, item in enumerate(row['levels']):
            count = item['valid_cells']
            levels.append(dict(level='P'+str(index+3),
                quality_gate=item['gate'].summary(), temporal_gain=item['gain'].summary(),
                internal_covariance=item['covariance'].summary(), valid_cells=count,
                nonzero_feature_correction_fraction=item['changed_cells']/count if count else None,
                feature_correction_RMS=math.sqrt(item['delta_squared_sum']/count) if count else None,
                relative_feature_correction_RMS=math.sqrt(item['delta_squared_sum']/max(item['base_squared_sum'], 1e-30)) if count else None,
                reliable_teacher_cells=item['reliable_cells'], reliable_changed_feature_cells=item['reliable_changed_cells']))
        return dict(frames=row['frames'], available_frames=row['available_frames'], missing_frames=row['missing_frames'],
            mode=row['mode'], quality_gate_enabled=row['quality_gate_enabled'],
            feature_protection_enabled=row['feature_protection_enabled'], levels=levels,
            scope='Read-only diagnostic after forward. Gate sample per available frame; gain/covariance weighted by valid feature cells. '
                'Missing inputs have no invented gate value. Feature correction is measured after dtype conversion; '
                'feature protection is not a guarantee of unchanged neighboring detector outputs. Internal covariance is not calibrated sensor noise.')

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
