"""Same-current-frame detector protocol, independently capped owned child."""
import os,socket,subprocess
from pathlib import Path
from .detector_runtime import DetectorRuntime,receive_arrays


class RawProfileDetectorRuntime(DetectorRuntime):
    def __init__(self,python,checkpoint,scripts_root):
        self.channel,peer=socket.socketpair();self.channel.settimeout(120)
        command=[str(python),'-u','tools/cfp_detector_worker_raw_profile.py','--fd',str(peer.fileno()),'--checkpoint',str(checkpoint),'--scripts-root',str(scripts_root)]
        self.process_handle=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,PYTHONPATH='.'),pass_fds=(peer.fileno(),));peer.close();_,self.metadata=receive_arrays(self.channel)
        expected=dict(torch='2.0.0+cu117',cuda='11.7',cudnn=8500)
        if any(self.metadata[k]!=v for k,v in expected.items()):self.close();raise ValueError('Pinned detector runtime mismatch')
