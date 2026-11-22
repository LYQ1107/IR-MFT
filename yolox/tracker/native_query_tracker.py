"""Past/current identity queries over a frozen RGB scorer and native cues.

The null alternative is an explicit possible answer. The additional encoder
cannot change the frozen RGB scorer, and zeroing its increment retains the
entire RGB real/null posterior. Previously returned identities are immutable.
"""
import json
from pathlib import Path

import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator, prepare_calibration_cue
from yolox.tracker.association_features import pair_features
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.native_query_features import identity_pair_features, posterior_with_null
from yolox.tracker.prebirth_identity_tracker import PrebirthIdentityMixin


class NativeQueryTracker(PrebirthIdentityMixin, CalibratedForegroundTracker):
    def __init__(self, config, model=None, native_model=None, prior=None):
        config = dict(config)
        checkpoint = None
        if native_model is None:
            checkpoint = torch.load(config['native_query_checkpoint'], map_location='cpu', weights_only=False)
            root = Path(__file__).resolve().parents[2]
            protocol = json.loads((root / 'configs/cfp/protocol_v2.json').read_text())
            if (checkpoint['fit_sequences'] != protocol['train'] or checkpoint['Dev_fit_rows']
                    or checkpoint['Test_fit_rows'] or checkpoint['feature_schema'] != 'native_query_v2_RGB26_cue9'):
                raise ValueError('Native query model violates the Train-only feature contract')
            from tools.run_pseudo_ir_comparison import sha256
            if checkpoint['config_sha256'] != sha256(root / 'configs/perceptual_native_query_v2_20261123.json'):
                raise ValueError('Native query training plan changed')
            if (config['association_checkpoint'] != checkpoint['core']['checkpoint']
                    or sha256(config['association_checkpoint']) != checkpoint['core']['checkpoint_sha256']):
                raise ValueError('Frozen RGB core mismatch')
            if checkpoint['frontend'] != config['detection_arm']:
                raise ValueError('Native query detector frontend mismatch')
            native_model = AssociationCalibrator(**checkpoint['architecture'])
            native_model.load_state_dict(checkpoint['model'], strict=True)
            prior = checkpoint['prior_parameters']
            self.native_cue = checkpoint['cue']
            self.native_metadata = {key: checkpoint[key] for key in [
                'cue', 'seed', 'epoch', 'config_sha256', 'feature_plan_sha256', 'sources',
                'core', 'prior_path', 'prior_sha256', 'prior_parameters', 'alignment_sha256']}
        else:
            self.native_cue = config.get('native_cue', 'ir')
            self.native_metadata = dict(scope='injected engineering-test model')
        if self.native_cue not in {'rgb', 'ir'} or prior is None:
            raise ValueError('Native cue/prior required')
        if config.get('native_cue', self.native_cue) != self.native_cue:
            raise ValueError('Native cue checkpoint mismatch')
        super().__init__(config, model=model, cue='rgb' if model is not None else None)
        if self.cue != 'rgb':
            raise ValueError('The complete ordinary RGB core must be retained')
        self.native_model = native_model.cpu().eval()
        self.prior = {key: float(prior[key]) for key in ['t', 'a', 'b', 'c']}
        self.native_cache = {}
        self.statistics.update(native_query_rows=0, native_increment_rows=0,
                               native_opened_edges=0, native_null_rejections=0)

    def update(self, frame, data):
        self.native_cache = {}
        return super().update(frame, data)

    def query_inputs(self, frame, context):
        """Use the complete current candidate set, never a filtered match stage."""
        data = self.data
        a, b = map(int, data['offsets'][frame-1:frame+1])
        index = int(context['index'])
        lag = frame - int(context['frame'])
        if lag <= 0 or index >= a:
            raise ValueError('Identity evidence must precede the current frame')
        point = data['foreground_transport']
        local = lag in self.config.get('lags', [1, 2, 4, 8])
        if local:
            c, d = map(int, data['offsets'][frame-lag-1:frame-lag+1])
            if not c <= index < d:
                raise ValueError('Past object index/frame mismatch')
        else:
            c, d = index, index + 1
        fields = ['boxes', 'embedding', 'rgb_mask_embedding', 'rgb_hog', 'color', 'quality']
        past = {key: data[key][c:d] for key in fields}
        current = {key: data[key][a:b] for key in fields}
        modes = ['rgb', 'bbox']
        if local:
            past['point_quality'] = point['quality'][c:d]
            current['point_quality'] = point['quality'][a:b]
            v0, v1 = map(int, point[f'vote_offsets_{lag}'][frame-1:frame+1])
            p0, p1 = map(int, point[f'count_offsets_{lag}'][frame-1:frame+1])
            matrices = {(s, t): point[f'vote_{lag}_{s}_{t}'][v0:v1].reshape(d-c, b-a)
                        for s, t in [('rgb', 'rgb'), ('rgb', 'bbox'), ('bbox', 'rgb')]}
            good = {s: point[f'good_{lag}_{s}'][p0:p1] for s in modes}
            total = {s: point[f'total_{lag}_{s}'][p0:p1] for s in modes}
        else:
            past['point_quality'] = np.zeros((1, 2), np.float32)
            current['point_quality'] = np.zeros((b-a, 2), np.float32)
            matrices = {(s, t): np.zeros((1, b-a), np.float32) for s in modes for t in modes}
            good = {s: np.zeros(1, np.float32) for s in modes}
            total = good
        base, extra, point_reference = pair_features(past, current, matrices, good, total, min(lag, 8), 'rgb')
        if not local:
            base[..., :3] = 0
            base[..., 12:14] = 0
            extra[..., 15:17] = 0
            point_reference[..., 15:17] = 0
        extra = prepare_calibration_cue(extra, 'rgb')
        native_enabled = not self.config.get('disable_native_increment', False)
        native_past = data['native_auxiliary'][c:d] if native_enabled else None
        native_current = data['native_auxiliary'][a:b] if native_enabled else None
        if native_enabled and self.config.get('shuffle_native_cue', False):
            native_current = np.roll(native_current, 1, axis=0)
        identity, cue7, reference7 = identity_pair_features(
            data['native_reference'][c:d], data['native_reference'][a:b], native_past, native_current)
        shape = (d-c, b-a)
        def qualities(which):
            return np.stack([np.broadcast_to(past['quality'][:, which:which+1], shape),
                             np.broadcast_to(current['quality'][None, :, which], shape)], axis=-1)
        rgb_quality = qualities(0)
        # A zero intervention does not access the IR quality or native arrays.
        cue_quality = qualities(1) if native_enabled and self.native_cue == 'ir' else rgb_quality
        available = (cue_quality[..., 0] >= .15) & (cue_quality[..., 1] >= .15) if self.native_cue == 'ir' else np.ones(shape, bool)
        return dict(start=c, base=np.concatenate([base, identity], -1), extra=extra,
                    point_reference=point_reference, cue=np.concatenate([cue7, cue_quality], -1),
                    reference=np.concatenate([reference7, rgb_quality], -1), available=available,
                    vote=matrices[('rgb', 'rgb')], good=good['rgb'], total=total['rgb'])

    def query_posterior(self, frame, context):
        lag = frame - int(context['frame'])
        key = ('local', lag) if lag in self.config.get('lags', [1, 2, 4, 8]) else ('long', int(context['index']))
        if key in self.native_cache:
            return self.native_cache[key]
        packet = self.query_inputs(frame, context)
        shape = packet['base'].shape[:2]
        n = int(np.prod(shape))
        core = np.zeros(n, np.float32)
        delta = np.zeros(n, np.float32)
        with torch.inference_mode():
            for lo in range(0, n, 4096):
                part = slice(lo, lo+4096)
                base = torch.from_numpy(packet['base'].reshape(-1, 26)[part])
                extra = torch.from_numpy(packet['extra'].reshape(-1, 17)[part])
                reference = torch.from_numpy(packet['point_reference'].reshape(-1, 17)[part])
                core[part] = self.model(base[:, :23], extra, reference, torch.ones(len(base), dtype=torch.bool)).numpy()
                if not self.config.get('disable_native_increment', False):
                    delta[part] = self.native_model.increment(
                        base, torch.from_numpy(packet['cue'].reshape(-1, 9)[part]),
                        torch.from_numpy(packet['reference'].reshape(-1, 9)[part]),
                        torch.from_numpy(packet['available'].reshape(-1)[part])).numpy()
        real = self.prior['t'] * core.reshape(shape) + self.prior['a'] * packet['base'][..., 23] + delta.reshape(shape)
        null = self.prior['b'] + self.prior['c'] * np.log1p(shape[1])
        probabilities, null_probability = posterior_with_null(real, null, self.config.get('query_null', True))
        packet.update(probability=probabilities, null=null_probability, delta=delta.reshape(shape))
        self.native_cache[key] = packet
        return packet

    def recovery_scores(self, frame, context, data):
        packet = self.query_posterior(frame, context)
        row = context['index'] - packet['start']
        probability = packet['probability'][row].copy()
        probability[probability <= packet['null'][row] + self.config.get('query_margin', .1)] = 0
        return probability, packet['delta'][row]

    def rank_cost(self, tracks, observations, base, frame, stage):
        result = super().rank_cost(tracks, observations, base, frame, stage)
        if not base.size or not self.config.get('query_ranking', True) or stage not in self.config.get('point_stages', ['primary', 'secondary']):
            return result
        columns = np.array([o.evidence['index']-self.current_start for o in observations], int)
        limit = {'primary': .9, 'secondary': .5, 'pending': .7}[stage]
        amount = float(self.config.get('query_weight', .8))
        for row, track in enumerate(tracks):
            history = track.object_observations if self.config.get('object_files', True) else [track.evidence]
            if not history:
                continue
            context = history[-1]
            packet = self.query_posterior(frame, context)
            index = context['index'] - packet['start']
            all_p = packet['probability'][index]
            if not len(all_p):
                continue
            best = int(all_p.argmax())
            maximum = float(all_p[best])
            second = float(np.partition(all_p, -2)[-2]) if len(all_p)>1 else 0.
            null = float(packet['null'][index])
            reliable_points = packet['good'][index] >= 4 and packet['vote'][index].max(initial=0) >= .15
            if (self.config.get('query_null', True) and null >= self.config.get('query_minimum_probability', .8)
                    and not reliable_points):
                result[row] = np.maximum(result[row], limit+1e-4)
                self.statistics['native_null_rejections'] += 1
                continue
            if (maximum < self.config.get('query_minimum_probability', .8)
                    or maximum-max(second, null) < self.config.get('query_margin', .1)
                    or self.data['scores'][self.current_start+best] < .3):
                continue
            confidence = maximum
            if best in columns:
                self.trusted[(track.track_id, self.current_start+best)] = confidence
            p = all_p[columns].astype(np.float64)
            scores = np.array([o.score for o in observations]) if stage != 'secondary' else np.ones(len(observations))
            weight = amount * confidence
            adjusted = (1-weight)*base[row] + weight*(1-p*scores)
            rejected = base[row] > limit
            support = ((packet['vote'][index, columns] >= .15)
                       | (packet['base'][index, columns, 23] >= .2)
                       | ((packet['cue'][index, columns, 0] >= .4)
                          & (packet['cue'][index, columns, 3:5].min(-1) >= .4)))
            opening = (p >= self.config.get('query_opening_probability', .95)) & support
            adjusted[rejected & ~opening] = result[row, rejected & ~opening]
            result[row] = adjusted
            self.statistics['native_query_rows'] += 1
            changed = abs(float(packet['delta'][index, best])) > 1e-4
            self.statistics['native_increment_rows'] += int(changed)
            self.statistics['IR_applied_rows'] += int(changed and self.native_cue == 'ir')
            self.statistics['native_opened_edges'] += int((rejected & (adjusted <= limit)).sum())
        return result
