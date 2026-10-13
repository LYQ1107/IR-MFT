"""Physical labels organize offline scenes, never student/online inputs."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256


class LocalSceneBank:
    def __init__(self, root, count=4, index_root=None, legacy_root=None):
        self.root, self.count = Path(root), count
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['status'] != 'COMPLETE_FROZEN_DINOV2_TWO_CROP_EVIDENCE' or count not in (4, 8):
            raise ValueError('Complete real DINO current-crop spatial evidence required')
        index, legacy = None, None
        if index_root is not None or legacy_root is not None:
            if index_root is None or legacy_root is None:
                raise ValueError('Native frame count and declared original sampling grid need both provenance roots')
            index_root, legacy_root = Path(index_root), Path(legacy_root)
            if (sha256(index_root/'manifest.json') != self.manifest['index_manifest_sha256']
                    or sha256(legacy_root/'manifest.json') != self.manifest['original_feature_manifest_sha256']):
                raise ValueError('Native-frame/sampling source changed')
            index = json.loads((index_root/'manifest.json').read_text())
            legacy = json.loads((legacy_root/'manifest.json').read_text())
        self.videos = {}
        for video, entry in self.manifest['videos'].items():
            directory = self.root/video
            if sha256(directory/'records.json') != entry['records_sha256']:
                raise ValueError('Physical observation record drift')
            arrays = {}
            for name in ('local'+str(count), 'visibility'+str(count), 'cls', 'valid'):
                path = directory/(name+'.npy')
                if sha256(path) != entry['artifacts'][name]['sha256']:
                    raise ValueError('Frozen spatial feature drift')
                arrays[name] = np.load(path, mmap_mode='r')
            records = json.loads((directory/'records.json').read_text())
            if index is not None:
                period = legacy['config']['frame_period']
                residues = legacy['config']['frame_residues']
                sampled_frames = [frame for frame in range(1, index['videos'][video]['frames']+1)
                                  if frame % period in residues]
                if any(row['frame'] not in sampled_frames for row in records):
                    raise ValueError('Observation not in the exact registered native-frame sampling grid')
            else:
                sampled_frames = sorted({row['frame'] for row in records})
            if (arrays['local'+str(count)].shape != (len(records), count, 384)
                    or arrays['visibility'+str(count)].shape != (len(records), count)
                    or arrays['cls'].shape != (len(records), 384)
                    or arrays['valid'].shape != (len(records),)):
                raise ValueError('Exact local/current shape alignment required')
            by_frame = {view: defaultdict(list) for view in ('gt', 'detector')}
            by_identity = {view: defaultdict(lambda: defaultdict(list)) for view in by_frame}
            for i, row in enumerate(records):
                by_frame[row['view']][row['frame']].append(i)
                if row['identity'] >= 0 and arrays['valid'][i]:
                    by_identity[row['view']][row['identity']][row['frame']].append(i)
            histories, ambiguous = {}, {}
            for view, mapping in by_identity.items():
                histories[view] = {identity: sorted((frame, rows[0]) for frame, rows in frames.items() if len(rows)==1)
                                   for identity, frames in mapping.items()}
                ambiguous[view] = sum(len(rows)!=1 for frames in mapping.values() for rows in frames.values())
            self.videos[video] = dict(records=records, arrays=arrays, frames=by_frame,
                identities=by_identity, histories=histories, ambiguous_history_frames=ambiguous,
                sampled_frames=sampled_frames,
                sampled_empty_both_frames=sum(not by_frame['gt'][frame] and not by_frame['detector'][frame]
                                             for frame in sampled_frames))

    def current_rows(self, video, indices):
        bank = self.videos[video]['arrays']
        ids = np.array(indices, dtype=np.int64)
        local = torch.from_numpy(np.array(bank['local'+str(self.count)][ids], dtype=np.float32))
        fraction = torch.from_numpy(np.array(bank['visibility'+str(self.count)][ids], dtype=np.float32))
        cls = torch.from_numpy(np.array(bank['cls'][ids], dtype=np.float32))
        valid = torch.from_numpy(np.array(bank['valid'][ids], dtype=np.bool_))
        fraction = fraction*valid[:, None]
        return local, fraction, cls

    def history_rows(self, video, identity, frame, length, maximum_age, view, rng=None, minimum_gap=1):
        source = self.videos[video]
        chosen = [(time, index) for time, index in source['histories']['gt' if view=='mixed' else view].get(identity, [])
                  if frame-maximum_age <= time <= frame-minimum_gap][-length:]
        if view == 'mixed':
            if rng is None:
                raise ValueError('Explicit sampling RNG required for declared training-only crop mixing')
            changed = []
            for time, index in chosen:
                detector = source['identities']['detector'].get(identity, {}).get(time, [])
                use_detector = rng.random() < .5 and len(detector)==1
                changed.append((time, detector[0] if use_detector else index))
            chosen = changed
        return chosen

    def scene(self, video, frame, identities, history_view='gt', candidate_view='detector',
              length=4, maximum_age=1500, rng=None, include_targets=False, minimum_gap=1):
        if not 1 <= minimum_gap <= maximum_age:
            raise ValueError('Declared positive past gap bounded by retained history age required')
        if history_view not in ('gt', 'detector', 'mixed') or candidate_view not in ('gt', 'detector'):
            raise ValueError('Declared clean/verified/mixed offline history and unchanged current view required')
        bank = self.videos[video]
        identities = list(identities)
        if len(set(identities)) != len(identities) or any(i<0 for i in identities):
            raise ValueError('Known unique physical offline query IDs required; labels never enter student')
        n = len(identities)
        local = torch.zeros(n, length, self.count, 384)
        fraction = torch.zeros(n, length, self.count)
        cls = torch.zeros(n, length, 384)
        times = torch.zeros(n, length, dtype=torch.int64)
        valid = torch.zeros(n, length, dtype=torch.bool)
        history_indices = []
        for j, identity in enumerate(identities):
            rows = self.history_rows(video, identity, frame, length, maximum_age, history_view, rng, minimum_gap)
            history_indices.append([index for time, index in rows])
            if rows:
                time_values, indices = zip(*rows)
                observed, support, global_values = self.current_rows(video, indices)
                local[j, -len(rows):], fraction[j, -len(rows):], cls[j, -len(rows):] = observed, support, global_values
                times[j, -len(rows):] = torch.tensor(time_values)
                valid[j, -len(rows):] = True
        current_indices = list(bank['frames'][candidate_view].get(frame, []))
        candidates, candidate_fraction, candidate_global = self.current_rows(video, current_indices)
        candidate_ids = torch.tensor([bank['records'][i]['identity'] for i in current_indices], dtype=torch.int64)
        model_inputs = dict(history=local, history_fraction=fraction, history_global=cls,
            times=times, history_valid=valid, frame=frame, candidates=candidates,
            candidate_fraction=candidate_fraction, candidate_global=candidate_global)
        result = dict(inputs=model_inputs, video=video, frame=frame,
            track_ids=torch.tensor(identities, dtype=torch.int64), candidate_ids=candidate_ids,
            history_indices=history_indices, current_indices=current_indices,
            minimum_gap=minimum_gap,
            scope='Offline clean physical grouping; not actual endogenous native track memory. Unknown proposals retained; no GT/IDs/target in model inputs.')
        if include_targets:
            target_indices = []
            for identity in identities:
                rows = bank['identities']['gt'].get(identity, {}).get(frame, [])
                if len(rows)!=1:
                    raise ValueError('Training clean current target must be independently unique/observed')
                target_indices.append(rows[0])
            a, b, c = self.current_rows(video, target_indices)
            result.update(target_local=a, target_fraction=b, target_global=c, target_indices=target_indices)
        return result

    def evaluation_ids(self, video, frame, maximum_age=1500):
        # Include past-observed fish that are currently absent, not only GT
        # current positives. This is still a declared clean offline inventory.
        mapping = self.videos[video]['histories']['gt']
        return sorted(identity for identity, rows in mapping.items()
                      if any(frame-maximum_age <= time < frame for time, index in rows))


class LocalSceneSampler:
    def __init__(self, bank, fit_videos, seed, identities_per_scene=8, length=4, maximum_age=1500, gaps=(1, 25, 100)):
        self.bank, self.videos = bank, list(fit_videos)
        self.rng = np.random.default_rng(seed)
        self.identities_per_scene, self.length, self.maximum_age = identities_per_scene, length, maximum_age
        self.gaps = tuple(gaps)
        self.labels, self.eligible = {}, {}
        keys = []
        for video in self.videos:
            entry = bank.videos[video]
            keys.extend((video, identity) for identity in entry['histories']['gt'])
            self.eligible[video] = {}
            for gap in self.gaps:
                frames = {}
                for frame in sorted(entry['frames']['gt']):
                    known = [identity for identity, rows in entry['histories']['gt'].items()
                             if len(entry['identities']['gt'][identity].get(frame, []))==1
                             and any(frame-maximum_age <= time <= frame-gap for time, index in rows)]
                    if len(known)>=2:
                        frames[frame] = known
                if not frames:
                    raise ValueError('Each fitting video/gap needs real same-frame different-fish and strictly past context')
                self.eligible[video][gap] = frames
        self.labels = {key: i for i, key in enumerate(sorted(keys))}

    def sample(self):
        video = self.videos[int(self.rng.integers(len(self.videos)))]
        gap = self.gaps[int(self.rng.integers(len(self.gaps)))]
        choices = sorted(self.eligible[video][gap])
        frame = choices[int(self.rng.integers(len(choices)))]
        ids = self.eligible[video][gap][frame]
        selected = self.rng.choice(ids, size=min(len(ids), self.identities_per_scene), replace=False).tolist()
        return self.bank.scene(video, frame, selected, 'mixed', 'detector', self.length,
                               self.maximum_age, self.rng, include_targets=True, minimum_gap=gap)
