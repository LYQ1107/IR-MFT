"""P3 ROI-coordinate contract for a future Dev-authorized F4 visual source.

No image encoding, learned model selection, feature bank or tracker mutation.
"""
import math

import torch
from torchvision.ops import roi_align


@torch.no_grad()
def pool_p3(feature,boxes_source,original_hw,input_hw,resize_ratio,normalize=True):
    """Pool source-image proposals after the explicit source->network transform.

    Returns FP32 embeddings and a per-ROI validity mask. Clipping affects only
    feature extraction, never the boxes subsequently passed to ByteTrack.
    """
    if feature.ndim!=4 or len(feature)!=1 or not feature.is_floating_point():
        raise ValueError('One current floating P3 feature map is required')
    if (len(original_hw)!=2 or len(input_hw)!=2 or
            any(not isinstance(x,int) or x<1 for x in [*original_hw,*input_hw])):
        raise ValueError('Explicit original and network input geometry required')
    expected=min(input_hw[0]/original_hw[0],input_hw[1]/original_hw[1])
    if not math.isfinite(resize_ratio) or not math.isclose(resize_ratio,expected,rel_tol=1e-7,abs_tol=1e-8):
        raise ValueError('ROI transform must use the actual aspect-preserving resize ratio')
    stride_y=input_hw[0]/feature.shape[-2];stride_x=input_hw[1]/feature.shape[-1]
    if stride_x!=stride_y or stride_x!=8:
        raise ValueError('The supplied map must be the actual stride8 P3 level')
    boxes=torch.as_tensor(boxes_source,device=feature.device,dtype=torch.float32).clone()
    if boxes.ndim!=2 or boxes.shape[1]!=4 or not bool(torch.isfinite(boxes).all()):
        raise ValueError('Finite original-image Nx4 proposal coordinates required')
    result=torch.zeros((len(boxes),feature.shape[1]),device=feature.device,dtype=torch.float32)
    if not len(boxes):return result,torch.zeros(0,device=feature.device,dtype=torch.bool)
    boxes*=resize_ratio
    valid_h,valid_w=int(original_hw[0]*resize_ratio),int(original_hw[1]*resize_ratio)
    boxes[:,0::2].clamp_(0,valid_w);boxes[:,1::2].clamp_(0,valid_h)
    valid=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    if bool(valid.any()):
        rois=torch.cat([torch.zeros((int(valid.sum()),1),device=feature.device),boxes[valid]],1)
        pooled=roi_align(feature.float(),rois,output_size=(3,3),spatial_scale=1/stride_x,
                         sampling_ratio=2,aligned=True).mean((-1,-2))
        if not bool(torch.isfinite(pooled).all()):raise ValueError('Learned ROI features are nonfinite')
        result[valid]=pooled
    norms=torch.linalg.vector_norm(result,dim=1)
    valid &= norms>1e-12
    if normalize:result[valid]/=norms[valid,None]
    result[~valid]=0
    return result,valid
