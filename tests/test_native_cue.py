import numpy as np
import torch

from yolox.tracker.native_cue import native_crop,native_input,same_video_triplet


def test_native_IR_preserves_scalar_pixels_and_zero_letterbox_padding():
    image=np.full((80,30),120,np.uint8)
    pixels,rectangle=native_crop(image,np.array([0,0,30,80]))
    value=native_input(torch.from_numpy(pixels[None]),torch.from_numpy(rectangle[None]),'cpu')
    assert value.shape==(1,3,128,256)
    assert torch.all(value[:,:,:,0]==0) and torch.all(value[:,:,:,-1]==0)
    dy,dx,rh,rw=rectangle
    decoded=value[0,:,dy+rh//2,dx+rw//2]*torch.tensor([.229,.224,.225])+torch.tensor([.485,.456,.406])
    torch.testing.assert_close(decoded,torch.full((3,),120/255),rtol=1e-6,atol=1e-6)


def test_triplet_requires_same_video_negative_and_sufficient_positive_time_gap():
    identity=torch.eye(4,requires_grad=True)
    labels=torch.tensor([0,0,1,2]);video=torch.tensor([0,0,1,1]);frame=torch.tensor([1,50,1,50])
    loss,fraction=same_video_triplet(identity,labels,video,frame)
    assert loss.item()==0 and fraction==0
    video[:]=0;loss,fraction=same_video_triplet(identity,labels,video,frame)
    assert fraction==.5 and torch.isfinite(loss)
    loss.backward();assert identity.grad is not None
