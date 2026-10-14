"""Current image/proposal-only excluded-video JPM encoder; no label lookup."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .dinov2_identity import load_pinned_small
from .dinov2_native_encoder import current_proposal_crops
from .transreid_dino_jpm import DinoJointPatchReID
from .masked_identity_checkpoint import load_delta_state
from .provenance import sha256


class CurrentJpmIdentityEncoder:
    def __init__(self, reference, weights, source_proof, training, deployment_video,
                 device='cuda', batch_size=128):
        if device not in ('cpu','cuda') or batch_size!=128 or training is None:
            raise ValueError('Completed excluded-video JPM fit and declared currentframepad128 required')
        training=Path(training);m=json.loads((training/'manifest.json').read_text());g=m['guard']
        if (m['status']!='COMPLETE' or deployment_video not in (g['inner'],g['outer'])
                or deployment_video in g['fit'] or g['inner']==g['outer'] or not m['heldout_BN_buffers_unchanged']
                or sha256(training/'final.pt')!=m['checkpoint_sha256'] or g['pretrained_sha256']!=sha256(weights)
                or any(sha256(p)!=h for p,h in g['source_sha256'].items())):
            raise ValueError('Exact completed model/source/coordinate and own excluded-video fold required')
        proofs=json.loads(Path(source_proof).read_text())['imported_source_proofs'];backbone=load_pinned_small(reference,weights,proofs)
        saved=torch.load(training/'final.pt',map_location='cpu',weights_only=False)
        if saved['guard']!=g or saved['epoch']!=m['config']['epochs']:raise ValueError('Exact fixedfinal checkpoint required')
        self.model=DinoJointPatchReID(backbone,len(m['classifier_keys']),g['mode']);load_delta_state(self.model,saved['model_delta'],self.model.immutable_base())
        self.model.eval().requires_grad_(False).to(device);self.device=device;self.batch_size=batch_size;self.dimension=m['output_dimension']
        self.metadata=dict(backbone_pretrained_sha256=sha256(weights),dimension=self.dimension,coordinate_sha256=m['checkpoint_sha256'],coordinate_kind='source_matched_DINO_JPM_'+g['mode']+'_after_BN',trained_encoder=dict(path=str(training),manifest_sha256=sha256(training/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],fit=g['fit'],inner=g['inner'],outer=g['outer'],seed=g['seed'],mode=g['mode']),encoder_batch_padding=128,encoder_precision='FP32/TF32disabled',pixel_quantization='NormalizedcropFP32->FP16->FP32, same registeredpilot',jpm_source=m['config']['source_adaptation'],scope='Currentcropafterfit-onlyBN descriptors; no scene/GT/future inputs, fixed originaldetectorcache proposals. SourceJPMparts includegloballymixedprefixCLS/tokens, notpurelocal/anatomical. Sourceadaptation notofficialwholeTransReID; no old512GRU coordinate reuse.')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

    def encode(self,rgb,boxes):
        started=time.perf_counter();crops,masks,valid=current_proposal_crops(rgb,boxes);timing=dict(current_crop_and_mask_seconds=time.perf_counter()-started,resize_transfer_seconds=0.,encoder_seconds=0.,feature_transfer_seconds=0.)
        output=dict(embedding=np.zeros((len(boxes),self.dimension),np.float32),valid=valid)
        with torch.inference_mode():
            for offset in range(0,len(crops),self.batch_size):
                count=min(self.batch_size,len(crops)-offset);started=time.perf_counter();padded=torch.zeros(self.batch_size,3,128,256);support=torch.zeros(self.batch_size,1,128,256);padded[:count]=crops[offset:offset+count];support[:count]=masks[offset:offset+count];images=padded.to(self.device);content=support.to(self.device)
                if self.device=='cuda':torch.cuda.synchronize()
                timing['resize_transfer_seconds']+=time.perf_counter()-started;started=time.perf_counter();observed=self.model(images,content)['embedding']
                if self.device=='cuda':torch.cuda.synchronize()
                timing['encoder_seconds']+=time.perf_counter()-started;started=time.perf_counter();current=observed[:count].float().cpu().numpy();current[~valid[offset:offset+count]]=0
                if not np.isfinite(current).all():raise ValueError('Nonfinite actual current JPM features')
                output['embedding'][offset:offset+count]=current;timing['feature_transfer_seconds']+=time.perf_counter()-started
        output['stage_seconds']=timing;return output
