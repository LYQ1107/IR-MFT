"""Read-only spatial/quality/redundancy diagnostics; no retained feature bank."""
import torch

from tools.native_pyramid_gate_observer import Moments


class SpatialObserver:
    def __init__(self):
        self.rows, self.handles = {}, []

    def attach(self, name, module):
        if name in self.rows:
            raise ValueError('One explicit diagnostic identity per module instance')
        row = dict(frames=0, available_frames=0, missing_frames=0, levels=[])
        for _ in range(3):
            row['levels'].append({key: Moments() for key in ['gate', 'frame_quality',
                'spatial_delta', 'redundancy', 'feature_delta', 'feature_changed']})
        self.rows[name] = row
        def hook(net, args, kwargs, output):
            with torch.no_grad():
                repaired, _, information = output
                row['frames'] += 1
                if information['missing_exact_RGB']:
                    row['missing_frames'] += 1
                    return
                row['available_frames'] += 1
                rgb = args[0] if args else kwargs['rgb']
                for item, fields, base, value in zip(row['levels'], information['levels'], rgb, repaired):
                    mask = fields['valid']
                    for key in ['gate', 'spatial_delta', 'redundancy']:
                        item[key].add(fields[key][mask])
                    item['frame_quality'].add(fields['frame_quality'])
                    delta = value.detach().float()-base.detach().float()
                    item['feature_delta'].add(delta.square().mean(1, keepdim=True)[mask].sqrt())
                    item['feature_changed'].add(delta.ne(0).any(1, keepdim=True)[mask].float())
        self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def summary(self, name):
        row = self.rows[name]
        return dict(frames=row['frames'], available_frames=row['available_frames'], missing_frames=row['missing_frames'],
            levels=[{k: v.summary() for k, v in item.items()} for item in row['levels']],
            scope='Gate/delta/rho valid spatial cells; frame quality once per frame. No invented values on missing cues. '
                'Read-only instrumented production is not individual system FPS; these proxies do not establish efficacy.')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
