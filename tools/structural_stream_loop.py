"""Bind live Train inputs to the unchanged, already used structural fit loop.

The function's bytecode, optimizer, loss, schedule and checkpoint logic stay
identical. Only its three input callbacks get private bindings. Importing this
module does not start fitting or change globals used by any running experiment.
"""
import hashlib
import json
import types

from tools import train_structural_relative as legacy


class StreamingLoopBridge:
    def __init__(self, provider, sequences, metadata):
        self.provider = provider
        self.sequences = list(sequences)
        if set(self.sequences) != set(provider.schedule) or len(self.sequences) != 6:
            raise ValueError('Require the same six fixed Train sequences')
        # Only lengths are consumed by the original batch generator. These are
        # address slots, not feature caches or fabricated target annotations.
        self.banks = {seq: {'targets': range(192)} for seq in self.sequences}
        self.structure = object()
        self.metadata = metadata
        metadata['live_frame_passes'] = []
        self.epoch = None
        self._pending = None
        self._active = False

    def batches(self, sequences, banks, seed, epoch):
        if self._active or list(sequences) != self.sequences or banks is not self.banks or seed != 0:
            raise ValueError('Unexpected stream, seed or concurrent input generator')
        self._active = True
        self.epoch = epoch
        self._pending = None
        started = self.provider.prepared
        digest = hashlib.sha256()
        used = {seq: set() for seq in self.sequences}
        self._digest, self._used = digest, used
        finished = False
        try:
            for draw in legacy.batches(sequences, banks, seed, epoch):
                if self._pending is not None:
                    raise RuntimeError('Previous feature batch was not consumed with its structure')
                yield draw
            finished = True
        finally:
            count = self.provider.prepared - started
            row = dict(pass_index=epoch, prepared_batches=count,
                       actual_frame_draw_sha256=digest.hexdigest(),
                       unique_frames={seq: len(v) for seq, v in used.items()},
                       complete_input_pass=bool(finished and count == 288 and
                                                all(len(v) == 192 for v in used.values())))
            self.metadata['live_frame_passes'].append(row)
            self._active = False
            if finished and not row['complete_input_pass']:
                # Raise before the legacy loop can save a nominally complete
                # pass or checkpoint. A deliberate short smoke closes early.
                raise RuntimeError('Full input pass did not deliver its registered coverage')

    def tensors(self, banks, draw, cue):
        if not self._active or banks is not self.banks or self._pending is not None:
            raise RuntimeError('Feature input requested out of sequence')
        batch = self.provider.prepare(draw, cue, self.epoch)
        expected = self.provider.addresses(draw, self.epoch)
        if batch.frames != expected:
            raise RuntimeError('Provider returned a different set of Train frames')
        for seq, frame in batch.frames:
            self._used[seq].add(frame)
        self._digest.update(json.dumps(batch.frames, separators=(',', ':')).encode())
        self._pending = (tuple(draw), cue, batch.structure)
        return batch.rgb, batch.auxiliary, batch.target

    def structural_tensors(self, structure, draw, cue):
        pending = self._pending
        if (structure is not self.structure or pending is None or
                pending[0] != tuple(draw) or pending[1] != cue):
            raise RuntimeError('Descriptors and feature input must come from the identical draw')
        self._pending = None
        return pending[2]

    def bind(self):
        namespace = dict(legacy.run_fit.__globals__)
        namespace.update(batches=self.batches, tensors=self.tensors,
                         structural_tensors=self.structural_tensors)
        function = types.FunctionType(legacy.run_fit.__code__, namespace,
                                      legacy.run_fit.__name__, legacy.run_fit.__defaults__,
                                      legacy.run_fit.__closure__)
        function.__kwdefaults__ = legacy.run_fit.__kwdefaults__
        assert function.__code__ is legacy.run_fit.__code__
        return function

    def run(self, args, adapter, head, optimizer, config, config_path, out):
        function = self.bind()
        function(args, adapter, head, optimizer, self.banks, self.structure,
                 self.sequences, config, config_path, out, self.metadata)
        rows = self.metadata['live_frame_passes']
        if args.smoke_steps:
            if len(rows) != 1 or rows[0]['prepared_batches'] != args.smoke_steps:
                raise RuntimeError('Incomplete diagnostic input delivery')
        elif len(rows) != config['epochs'] or not all(r['complete_input_pass'] for r in rows):
            raise RuntimeError('Incomplete registered Train input coverage')
