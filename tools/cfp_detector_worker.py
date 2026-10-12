#!/usr/bin/env python3
"""Receive only the current pair, return frozen V3 proposals, never read GT."""
import argparse
import socket

import cv2
import torch

from cfp.detector_runtime import receive_arrays, send_arrays
from cfp.online import CurrentFrameFrontend


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fd', type=int, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--scripts-root', required=True)
    p.add_argument('--batch', type=int, choices=[1, 8], required=True)
    p.add_argument('--preprocessing', choices=['original', 'lookup_uint8'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if torch.cuda.device_count() != 1:
        raise ValueError('One explicitly visible detector GPU is required')
    torch.manual_seed(0)
    frontend = CurrentFrameFrontend(args.checkpoint, args.scripts_root, detector_batch=args.batch,
                                    input_regime='causal_precomputed', preprocessing=args.preprocessing)
    channel = socket.socket(fileno=args.fd)
    send_arrays(channel, {}, dict(torch=torch.__version__, cuda=torch.version.cuda,
                                 cudnn=torch.backends.cudnn.version(), parameters=frontend.detector_parameters,
                                 device=torch.cuda.get_device_name()))
    frames = 0
    try:
        while True:
            arrays, metadata = receive_arrays(channel)
            if metadata['action'] == 'stop':
                send_arrays(channel, {}, dict(frames_processed=frames, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))
                break
            if metadata['action'] != 'frame':
                raise ValueError('Unknown current-frame detector request')
            result = frontend.process(arrays['rgb'], provided_ir=arrays['ir'])
            frames += 1
            send_arrays(channel, {k: result[k] for k in ['boxes', 'scores', 'raw_post_nms']}, dict(stage_seconds=result['stage_seconds']))
    except EOFError:
        pass
    finally:
        channel.close()


if __name__ == '__main__':
    main()
