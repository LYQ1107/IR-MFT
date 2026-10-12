"""Private current-frame IPC for separately pinned detector/encoder runtimes."""
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess

import numpy as np


def send_arrays(channel, arrays, metadata):
    stream = io.BytesIO()
    np.savez(stream, **arrays, metadata=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), np.uint8))
    data = stream.getvalue()
    if len(data) > 64 * 1024 * 1024:
        raise ValueError('Current-frame IPC payload exceeds declared bound')
    channel.sendall(struct.pack('!Q', len(data)) + data)


def receive_arrays(channel):
    def read(size):
        parts = []
        while size:
            part = channel.recv(min(size, 4 * 1024 * 1024))
            if not part:
                raise EOFError('Detector worker closed its private stream')
            parts.append(part)
            size -= len(part)
        return b''.join(parts)
    size = struct.unpack('!Q', read(8))[0]
    if size > 64 * 1024 * 1024:
        raise ValueError('Detector worker response exceeds declared bound')
    with np.load(io.BytesIO(read(size)), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != 'metadata'}
        metadata = json.loads(archive['metadata'].tobytes())
    if metadata.get('error'):
        raise RuntimeError(metadata['error'])
    return arrays, metadata


class DetectorRuntime:
    def __init__(self, python, checkpoint, scripts_root, batch, preprocessing):
        self.channel, peer = socket.socketpair()
        self.channel.settimeout(120)
        command = [str(python), '-u', 'tools/cfp_detector_worker.py', '--fd', str(peer.fileno()),
                   '--checkpoint', str(checkpoint), '--scripts-root', str(scripts_root),
                   '--batch', str(batch), '--preprocessing', preprocessing]
        self.process_handle = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                                env=dict(os.environ, PYTHONPATH='.'), pass_fds=(peer.fileno(),))
        peer.close()
        _, self.metadata = receive_arrays(self.channel)
        expected = dict(torch='2.0.0+cu117', cuda='11.7', cudnn=8500)
        if any(self.metadata[k] != value for k, value in expected.items()):
            self.close()
            raise ValueError('Detector worker does not match the frozen cu117 runtime')

    def process(self, rgb, ir):
        send_arrays(self.channel, dict(rgb=rgb, ir=ir), dict(action='frame'))
        arrays, metadata = receive_arrays(self.channel)
        return arrays, metadata['stage_seconds']

    def close(self):
        if self.process_handle.poll() is None:
            try:
                send_arrays(self.channel, {}, dict(action='stop'))
                _, summary = receive_arrays(self.channel)
                self.metadata.update(summary)
                try:
                    self.process_handle.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # All frame work and the stop acknowledgement are complete.
                    # CUDA context teardown can exceed the grace period; retain
                    # worker statistics before cleaning up this owned child.
                    self.metadata['shutdown_grace_timeout_seconds'] = 10
            finally:
                self.channel.close()
                if self.process_handle.poll() is None:
                    self.process_handle.terminate()
                    self.metadata['shutdown_owned_child_terminated'] = True
                    try:
                        self.process_handle.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process_handle.kill()
                        self.metadata['shutdown_owned_child_killed'] = True
                        self.process_handle.wait(timeout=10)
        self.metadata['shutdown_returncode'] = self.process_handle.poll()
        return self.metadata
