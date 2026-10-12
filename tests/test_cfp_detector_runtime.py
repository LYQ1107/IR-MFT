import socket
import struct

import numpy as np
import pytest

from cfp.detector_runtime import send_arrays, receive_arrays


def test_private_runtime_transport_preserves_current_pixels_and_proposal_precision():
    source = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    boxes = np.array([[.125, 2., 31.375, 80.]], np.float32)
    left, right = socket.socketpair()
    try:
        send_arrays(left, dict(rgb=source, boxes=boxes), dict(action='frame', stage_seconds={'detector': .125}))
        arrays, metadata = receive_arrays(right)
        assert np.array_equal(arrays['rgb'], source) and arrays['rgb'].dtype == np.uint8
        assert np.array_equal(arrays['boxes'], boxes) and arrays['boxes'].dtype == np.float32
        assert metadata == {'action': 'frame', 'stage_seconds': {'detector': .125}}
    finally:
        left.close()
        right.close()


def test_private_runtime_refuses_oversized_or_closed_streams():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack('!Q', 64 * 1024 * 1024 + 1))
        with pytest.raises(ValueError, match='exceeds declared bound'):
            receive_arrays(right)
        left.close()
        with pytest.raises(EOFError, match='closed its private stream'):
            receive_arrays(right)
    finally:
        left.close()
        right.close()
