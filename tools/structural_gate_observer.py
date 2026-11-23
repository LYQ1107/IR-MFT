"""Read-only feature-risk/activation statistics; no labels or altered outputs."""
import torch


class StructuralGateObserver:
    def __init__(self):
        self.buffers = {}
        self.handles = []

    def attach(self, name, adapter):
        for index, layer in enumerate(adapter.levels):
            def observe(module, args, output, level=index+3, model_name=name):
                self.record(model_name, level, module, args, output)
            self.handles.append(layer.register_forward_hook(observe))

    @torch.no_grad()
    def record(self, name, level, module, args, output):
        assert len(args) == 9
        rgb, _, teacher_score = args[:3]
        availability, null = args[7:9]
        probability = teacher_score.detach().float().clamp(0, 1)
        if isinstance(availability, bool):
            valid = torch.full_like(probability, availability, dtype=torch.bool)
        else:
            valid = availability.to(device=probability.device, dtype=torch.bool).reshape(-1,1,1,1).expand_as(probability)
        risk = (1-probability)*(probability < module.risk_threshold)*valid
        delta = output.detach().float()-rgb.detach().float()
        active = delta.abs().amax(1,keepdim=True) > 0
        correction = delta.square().mean(1,keepdim=True).sqrt()/rgb.detach().float().square().mean(1,keepdim=True).add(1e-6).sqrt()
        key = (name, level, 'null_reference' if null else 'current')
        sums = dict(valid=valid.sum(), eligible=(risk > 0).sum(), risk=risk.sum(dtype=torch.float64),
                    active=active.sum(), correction=correction.sum(dtype=torch.float64))
        if key not in self.buffers:
            self.buffers[key] = dict(frames=0,positions=0,calls=0,sums={k:torch.zeros_like(v) for k,v in sums.items()},
                                     peak=torch.zeros((),device=correction.device,dtype=correction.dtype))
        row = self.buffers[key]
        row['frames'] += int(len(rgb)); row['positions'] += int(probability.numel()); row['calls'] += 1
        for key, value in sums.items():
            row['sums'][key].add_(value)
        row['peak'] = torch.maximum(row['peak'], correction.amax())

    def summary(self, name):
        rows = []
        for (model_name, level, branch), record in sorted(self.buffers.items()):
            if model_name != name:
                continue
            n = record['positions']; sums = {key:float(value.item()) for key,value in record['sums'].items()}
            rows.append(dict(level=level,branch=branch,frames=record['frames'],positions=n,calls=record['calls'],
                available_fraction=sums['valid']/n,teacher_risk_eligible_fraction=sums['eligible']/n,
                mean_teacher_risk_factor=sums['risk']/n,nonzero_feature_correction_fraction=sums['active']/n,
                mean_feature_correction_in_RGB_RMS=sums['correction']/n,
                maximum_feature_correction_in_RGB_RMS=float(record['peak'].item())))
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


SCOPE = ('Read-only forward hooks on primary no-cycle/full-cycle and matched RGB/gray or IR intervention models. '
         'Teacher risk is exactly (1-p)*(p<.6) times availability; it excludes structural-cost confidence and tanh. '
         'Feature activation/RMS measures actual post-rounding feature changes, not true correspondence, correct '
         'detections or HOTA. Null-reference statistics describe its computed branch. No GT, gradients or future inputs. '
         'Counters remain on device until the complete producer finishes. Hook overhead is included only in bundled '
         'diagnostic production; separate per-method runtime profiling does not attach these hooks.')
