"""Bounded original-image RAM cache with byte identity across metadata changes.

SSHFS inode identities are metadata, not an image-content checksum. A changed
fingerprint triggers a fresh read and exact comparison, never silent reuse of
changed pixels. Digests survive cache eviction. No image/FPN disk cache.
"""
import hashlib
import json

import cv2
import numpy as np

from tools.structural_train_image_reader import TrainImageReader, available_memory_bytes


class ContentVerifiedTrainImageReader(TrainImageReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_hashes = {}
        self.metadata_only_revalidations = 0
        self.double_read_checks = 0

    @staticmethod
    def fingerprint(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    def stable_bytes(self, path):
        before = self.fingerprint(path.stat())
        value = path.read_bytes()
        after = self.fingerprint(path.stat())
        if before != after:
            second = path.read_bytes()
            self.double_read_checks += 1
            if second != value:
                raise RuntimeError(f'Original Train image bytes changed while reading: {path}')
            after = self.fingerprint(path.stat())
        return after, value

    def read(self, sequence, frame, modality):
        if (sequence not in self.sequences or '/' in sequence or '\\' in sequence or
                int(frame) != frame or frame < 1 or modality not in ['rgb', 'ir']):
            raise ValueError('Only registered Train frames and RGB/IR are allowed')
        root, folder, suffix, flag = ((self.rgb_root, 'visible', 'jpg', cv2.IMREAD_COLOR)
            if modality == 'rgb' else (self.ir_root, 'infrared_diffv2ir', 'png', cv2.IMREAD_UNCHANGED))
        path = root / 'train' / sequence / folder / f'{int(frame):06d}.{suffix}'
        key = sequence, int(frame), modality
        current = self.fingerprint(path.stat())
        row = self.cache.get(key)
        if row is not None:
            value = row[1]
            if row[0] != current:
                current, fresh = self.stable_bytes(path)
                if fresh != value:
                    raise RuntimeError(f'Original Train image bytes changed during RAM reuse: {path}; '
                        f'cached_SHA256={hashlib.sha256(value).hexdigest()}; '
                        f'current_SHA256={hashlib.sha256(fresh).hexdigest()}')
                self.cache[key] = current, value
                self.metadata_only_revalidations += 1
            self.cache.move_to_end(key)
            self.hits += 1
        else:
            current, value = self.stable_bytes(path)
            digest = hashlib.sha256(value).hexdigest()
            expected = self.content_hashes.get(key)
            if expected is not None and expected['sha256'] != digest:
                raise RuntimeError(f'Original Train image bytes changed after cache eviction: {path}; '
                    f'first_SHA256={expected["sha256"]}; current_SHA256={digest}')
            self.content_hashes[key] = dict(sha256=digest, bytes=len(value))
            self.misses += 1
            length = len(value)
            if length <= self.max_bytes and available_memory_bytes()-length >= self.reserve_bytes:
                while self.cache and self.bytes+length > self.max_bytes:
                    old = self.cache.popitem(last=False)[1][1]
                    self.bytes -= len(old)
                    self.evictions += 1
                self.cache[key] = current, value
                self.bytes += length
        decoded = cv2.imdecode(np.frombuffer(value, np.uint8), flag)
        if decoded is None:
            raise ValueError('Required original Train image could not be decoded')
        if modality == 'ir' and (decoded.ndim != 2 or decoded.dtype != np.uint8):
            raise ValueError('Actual November pseudo-IR must remain scalar uint8')
        return decoded

    def content_manifest(self):
        return [dict(sequence=seq, frame=frame, modality=modality, **info)
                for (seq, frame, modality), info in sorted(self.content_hashes.items())]

    def statistics(self):
        values = super().statistics()
        records = self.content_manifest()
        values.update(content_SHA256_entries=len(records),
            content_manifest_sha256=hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            metadata_only_revalidations=self.metadata_only_revalidations,
            double_read_checks=self.double_read_checks, content_guard_survives_eviction=True)
        return values
