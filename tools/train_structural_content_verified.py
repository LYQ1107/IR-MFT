#!/usr/bin/env python3
"""Unchanged formal fitter, privately bound to a content-verified input reader."""
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_structural_full_train as original
from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.run_pseudo_ir_comparison import dump, sha256


class TrackedBridge(original.FullTrainLoopBridge):
    def __init__(self, provider, sequences, metadata, table):
        super().__init__(provider, sequences, metadata, table)
        self.reader = provider.image_reader
        metadata['source_hashes'].update({name: sha256(ROOT / name) for name in [
            'tools/content_verified_train_reader.py', 'tools/train_structural_content_verified.py']})
        metadata['content_guard_scope'] = ('Metadata changes require exact current-vs-cached byte identity; '
            'first-read SHA256 also survives eviction. No changed image is accepted. '
            'Original loss/optimizer/address schedule and fitter bytecode unchanged.')

    def batches(self, *args):
        try:
            yield from super().batches(*args)
        finally:
            self.metadata['content_guard_statistics'] = self.reader.statistics()

    def run(self, args, adapter, head, optimizer, config, config_path, out):
        super().run(args, adapter, head, optimizer, config, config_path, out)
        manifest = out / 'source_image_content_manifest.json'
        assert not manifest.exists()
        dump(manifest, self.reader.content_manifest())
        path = out / 'complete.json'
        value = json.loads(path.read_text())
        value['content_guard_statistics'] = self.reader.statistics()
        value['source_image_content_manifest'] = dict(path=str(manifest), sha256=sha256(manifest))
        dump(path, value)


def main():
    namespace = dict(original.main.__globals__)
    namespace.update(TrainImageReader=ContentVerifiedTrainImageReader, FullTrainLoopBridge=TrackedBridge)
    bound = types.FunctionType(original.main.__code__, namespace, original.main.__name__,
        original.main.__defaults__, original.main.__closure__)
    assert bound.__code__ is original.main.__code__
    bound()


if __name__ == '__main__':
    main()
