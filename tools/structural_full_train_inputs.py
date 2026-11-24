"""Use every registered Train frame without building new image/FPN disk caches."""
import hashlib
import json

import numpy as np

from tools.structural_stream_loop import StreamingLoopBridge
from tools.structural_stream_training import StreamingTrainInputs


class FullTrainInputs(StreamingTrainInputs):
    def __init__(self, *args, lengths, **kwargs):
        super().__init__(*args, **kwargs)
        if set(lengths) != set(self.protocol['train']):
            raise ValueError('Only the fixed six Train video lengths are allowed')
        self.lengths = dict(lengths)

    def addresses(self, draw, epoch):
        if not 1 <= epoch <= self.config['epochs']:
            raise ValueError('Unregistered optimizer macro pass')
        result = []
        for sequence, frame in draw:
            if sequence not in self.lengths or int(frame) != frame or not 1 <= frame <= self.lengths[sequence]:
                raise ValueError('Frame is outside fixed Train video bounds')
            result.append((sequence, int(frame)))
        return result


class FullTrainLoopBridge(StreamingLoopBridge):
    def __init__(self, provider, sequences, metadata, table):
        super().__init__(provider, sequences, metadata)
        self.table = np.asarray(table)
        if self.table.shape != (provider.config['epochs'], 288, 4, 2):
            raise ValueError('Require registered batch4,288update macro passes')
        if self.table.dtype.kind not in 'iu' or self.table[..., 0].min() < 0 or self.table[..., 0].max() >= 6:
            raise ValueError('Invalid Train sequence addresses')
        for index, sequence in enumerate(sequences):
            frames = self.table[..., 1][self.table[..., 0] == index]
            if not len(frames) or frames.min() < 1 or frames.max() > provider.lengths[sequence]:
                raise ValueError('Invalid Train frame addresses')
        pairs = self.table.reshape(-1, 2, 2)
        if np.any(pairs[:, 0, 0] != pairs[:, 1, 0]) or np.any(pairs[:, 0, 1] == pairs[:, 1, 1]):
            raise ValueError('Counterfactual pairs must be distinct frames of the same video')

    def batches(self, sequences, banks, seed, epoch):
        if (self._active or list(sequences) != self.sequences or banks is not self.banks or
                seed != 0 or not 1 <= epoch <= len(self.table)):
            raise ValueError('Unexpected stream, seed or macro pass')
        self._active, self.epoch, self._pending = True, epoch, None
        self._digest, self._used = hashlib.sha256(), {seq: set() for seq in self.sequences}
        started = self.provider.prepared
        expected = hashlib.sha256()
        finished = False
        try:
            for row in self.table[epoch - 1]:
                if self._pending is not None:
                    raise RuntimeError('Previous input batch was not consumed with its descriptors')
                draw = [(self.sequences[int(index)], int(frame)) for index, frame in row]
                expected.update(json.dumps(draw, separators=(',', ':')).encode())
                yield draw
            finished = True
        finally:
            count = self.provider.prepared - started
            complete = finished and count == 288 and self._digest.hexdigest() == expected.hexdigest()
            self.metadata['live_frame_passes'].append(dict(pass_index=epoch,
                logical_Train_epoch=(epoch - 1) // 24 + 1, prepared_batches=count,
                actual_frame_draw_sha256=self._digest.hexdigest(),
                expected_frame_draw_sha256=expected.hexdigest(),
                unique_frames={seq: len(v) for seq, v in self._used.items()},
                complete_input_pass=complete))
            self._active = False
            if finished and not complete:
                raise RuntimeError('Full macro pass failed registered frame delivery')
