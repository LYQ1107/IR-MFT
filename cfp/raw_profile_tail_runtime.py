"""Same-current-frame detector protocol, independently capped owned child."""
import os,socket,subprocess
from pathlib import Path
from .detector_runtime import DetectorRuntime,receive_arrays,send_arrays


class RawProfileTailValidatedRuntime(DetectorRuntime):
    def __init__(self,python,checkpoint,scripts_root):
        self.channel,peer=socket.socketpair();self.channel.settimeout(120)
        command=[str(python),'-u','tools/cfp_detector_worker_raw_profile_v2.py','--fd',str(peer.fileno()),'--checkpoint',str(checkpoint),'--scripts-root',str(scripts_root)]
        self.process_handle=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,PYTHONPATH='.'),pass_fds=(peer.fileno(),));peer.close();_,self.metadata=receive_arrays(self.channel)
        expected=dict(torch='2.0.0+cu117',cuda='11.7',cudnn=8500)
        if any(self.metadata[k]!=v for k,v in expected.items()):self.close();raise ValueError('Pinned detector runtime mismatch')

    def validate_tail(self,rgb,ir,size,position):
        send_arrays(self.channel,dict(rgb=rgb,ir=ir),dict(action='cache_tail_validation',size=size,position=position))
        arrays,metadata=receive_arrays(self.channel)
        if not metadata.get('validation_only') or not metadata.get('current_only_zeros_other_positions'):raise ValueError('Current-only tail validation required')
        return arrays
