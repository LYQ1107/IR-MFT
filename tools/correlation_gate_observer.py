"""Read-only CGR learned gate and post-dtype feature-correction statistics."""
import math

import torch

from tools.native_pyramid_gate_observer import Moments


class CorrelationGateObserver:
    def __init__(self):
        self.rows, self.handles, self.pending = {}, [], {}

    def attach(self, name, adapter):
        if name in self.rows:
            raise ValueError('Unique observer case ID required')
        self.rows[name] = []
        for index, layer in enumerate(adapter.levels):
            row = dict(frames=0, available_frames=0, cells=0, changed=0,
                reliable=0, reliable_changed=0, delta2=0., base2=0.,
                learned_gate=Moments(), effective_trust=Moments())
            self.rows[name].append(row)
            key = name, index
            self.pending[key] = None

            def gate_hook(module, args, output, slot=key):
                self.pending[slot] = output.detach()

            def level_hook(module, args, output, slot=key, item=row):
                with torch.no_grad():
                    logits, self.pending[slot] = self.pending[slot], None
                    rgb, _, score, availability = args
                    if availability is None:
                        flags = torch.ones(len(rgb), device=rgb.device, dtype=torch.bool)
                    elif isinstance(availability, bool):
                        flags = torch.full((len(rgb),), availability, device=rgb.device, dtype=torch.bool)
                    else:
                        flags = availability.to(device=rgb.device, dtype=torch.bool)
                    mask = flags.reshape(-1, 1, 1, 1).expand_as(rgb[:, :1])
                    item['frames'] += len(rgb)
                    item['available_frames'] += int(flags.sum())
                    if not bool(flags.any()):
                        assert logits is None
                        assert torch.equal(rgb, output)
                        return
                    probability = (torch.zeros_like(rgb[:, :1]).float() if score is None
                                   else score.detach().float().clamp(0, 1))
                    if module.risk_gate:
                        assert logits is not None
                        learned = logits.sigmoid()
                        trust = learned * (1-probability) * (probability < .6)
                        item['learned_gate'].add(learned[mask])
                        item['effective_trust'].add(trust[mask])
                    changed = (output.detach().float() != rgb.detach().float()).any(1, keepdim=True)
                    reliable = mask & (probability >= .6)
                    item['cells'] += int(mask.sum())
                    item['changed'] += int((changed & mask).sum())
                    item['reliable'] += int(reliable.sum())
                    item['reliable_changed'] += int((changed & reliable).sum())
                    delta2 = (output.detach().float()-rgb.detach().float()).square().mean(1, keepdim=True)
                    base2 = rgb.detach().float().square().mean(1, keepdim=True)
                    item['delta2'] += float(delta2[mask].double().sum())
                    item['base2'] += float(base2[mask].double().sum())
            self.handles.append(layer.gate.register_forward_hook(gate_hook))
            self.handles.append(layer.register_forward_hook(level_hook))

    def summary(self, name):
        result = []
        for index, row in enumerate(self.rows[name]):
            count = row['cells']
            result.append(dict(level='P'+str(index+3), frames=row['frames'],
                available_frames=row['available_frames'], missing_frames=row['frames']-row['available_frames'],
                available_feature_cells=count, learned_sigmoid_gate=row['learned_gate'].summary(),
                effective_teacher_masked_trust=row['effective_trust'].summary(),
                nonzero_feature_correction_fraction=row['changed']/count if count else None,
                feature_correction_RMS=math.sqrt(row['delta2']/count) if count else None,
                relative_feature_correction_RMS=math.sqrt(row['delta2']/max(row['base2'], 1e-30)) if count else None,
                reliable_teacher_cells=row['reliable'], reliable_changed_feature_cells=row['reliable_changed']))
        assert all(value is None for key, value in self.pending.items() if key[0] == name)
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()


SCOPE = ('Read-only original CorrelationRepair hooks; no labels or changed outputs. Learned sigmoid gate '
    'and (1-p)*(p<.6)-masked trust use available feature cells including padding, not GT object regions. '
    'Missing frames do not fabricate gate samples. Corrections are measured after the actual output dtype '
    'cast; reliable feature preservation is not a guarantee of unchanged neighboring detector boxes. '
    'Temporary gate logits are cleared after each layer. Observer overhead must be excluded from '
    'independent per-method timing and disclosed in bundled production.')
