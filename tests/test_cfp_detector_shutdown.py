import subprocess
import pytest
from cfp.detector_runtime import DetectorRuntime


class Process:
    def __init__(self):self.returncode=None;self.waits=0;self.terminated=False
    def poll(self):return self.returncode
    def wait(self,timeout):
        self.waits+=1
        if self.waits==1:raise subprocess.TimeoutExpired('owned-worker',timeout)
        self.returncode=-15;return self.returncode
    def terminate(self):self.terminated=True
    def kill(self):raise AssertionError('Graceful termination should suffice')


class Channel:
    def __init__(self):self.closed=False
    def close(self):self.closed=True


def test_acknowledged_frame_statistics_survive_owned_worker_shutdown_timeout(monkeypatch):
    runtime=DetectorRuntime.__new__(DetectorRuntime)
    runtime.process_handle=Process();runtime.channel=Channel();runtime.metadata={}
    monkeypatch.setattr('cfp.detector_runtime.send_arrays',lambda channel,arrays,metadata:None)
    monkeypatch.setattr('cfp.detector_runtime.receive_arrays',lambda channel:({},dict(frames_processed=4834,peak_gpu_memory_bytes=123)))
    stats=runtime.close()
    assert stats['frames_processed']==4834 and stats['peak_gpu_memory_bytes']==123
    assert stats['shutdown_grace_timeout_seconds']==10
    assert runtime.process_handle.terminated and runtime.channel.closed


def test_processing_or_receive_errors_are_not_converted_to_success(monkeypatch):
    runtime=DetectorRuntime.__new__(DetectorRuntime)
    runtime.process_handle=Process();runtime.process_handle.waits=1
    runtime.channel=Channel();runtime.metadata={}
    monkeypatch.setattr('cfp.detector_runtime.send_arrays',lambda channel,arrays,metadata:None)
    def failure(channel):raise EOFError('No stop acknowledgement')
    monkeypatch.setattr('cfp.detector_runtime.receive_arrays',failure)
    with pytest.raises(EOFError):runtime.close()
    assert runtime.channel.closed and runtime.process_handle.terminated
