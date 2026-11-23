"""Bounded encoded-byte RAM reuse for original Train images; no disk cache."""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


def available_memory_bytes():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot verify shared-host available memory')


class TrainImageReader:
    def __init__(self, rgb_root, ir_root, sequences, max_bytes=8 * 2**30,
                 reserve_bytes=40 * 2**30):
        self.rgb_root, self.ir_root = Path(rgb_root), Path(ir_root)
        self.sequences = frozenset(sequences)
        if not self.sequences or max_bytes < 0 or reserve_bytes < 0:
            raise ValueError('Require Train sequences and nonnegative RAM limits')
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes
        self.cache = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def read(self, sequence, frame, modality):
        if (sequence not in self.sequences or '/' in sequence or '\\' in sequence or
                int(frame) != frame or frame < 1 or modality not in ['rgb', 'ir']):
            raise ValueError('Only registered Train frames and RGB/IR are allowed')
        root, folder, suffix, flag = ((self.rgb_root, 'visible', 'jpg', cv2.IMREAD_COLOR)
                                     if modality == 'rgb' else
                                     (self.ir_root, 'infrared_diffv2ir', 'png', cv2.IMREAD_UNCHANGED))
        path = root / 'train' / sequence / folder / f'{int(frame):06d}.{suffix}'
        info = path.stat()
        fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        key = (sequence, int(frame), modality)
        row = self.cache.get(key)
        if row is not None:
            if row[0] != fingerprint:
                raise RuntimeError('Original Train image changed during RAM reuse')
            self.cache.move_to_end(key)
            value = row[1]
            self.hits += 1
        else:
            value = path.read_bytes()
            after = path.stat()
            if fingerprint != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError('Original Train image changed while reading')
            self.misses += 1
            length = len(value)
            available = available_memory_bytes()
            if length <= self.max_bytes and available - length >= self.reserve_bytes:
                while self.cache and self.bytes + length > self.max_bytes:
                    old = self.cache.popitem(last=False)[1][1]
                    self.bytes -= len(old)
                    self.evictions += 1
                self.cache[key] = (fingerprint, value)
                self.bytes += length
        decoded = cv2.imdecode(np.frombuffer(value, np.uint8), flag)
        if decoded is None:
            raise ValueError('Required original Train image could not be decoded')
        if modality == 'ir' and (decoded.ndim != 2 or decoded.dtype != np.uint8):
            raise ValueError('Actual November pseudo-IR must remain scalar uint8')
        return decoded

    def statistics(self):
        return dict(encoded_RAM_bytes=self.bytes, limit_bytes=self.max_bytes,
                    available_memory_reserve_bytes=self.reserve_bytes,
                    hits=self.hits, misses=self.misses, evictions=self.evictions,
                    entries=len(self.cache), image_or_feature_disk_cache=False)


def with_image_reader(provider_class):
    """An explicit subclass keeps the previously checked input helper unchanged."""
    class ReusingTrainInputs(provider_class):
        def __init__(self, *args, image_reader, **kwargs):
            super().__init__(*args, **kwargs)
            if image_reader.sequences != frozenset(self.protocol['train']):
                raise ValueError('Image reader and provider Train sequence sets differ')
            self.image_reader = image_reader

        def _images(self, sequence, frame, cue):
            rgb = self.image_reader.read(sequence, frame, 'rgb')
            ir = self.image_reader.read(sequence, frame, 'ir') if cue == 'ir' else None
            if ir is not None and ir.shape != rgb.shape[:2]:
                raise ValueError('Original RGB/IR image geometry differs')
            return rgb, ir

    return ReusingTrainInputs
