# Train-input and optimizer engineering parity

The CPU-only diagnostic completed on8actualcached Trainframes from BT-001,
BT-003,MSK-002 and SN-015. It used the first-pass addresses already fixed
in the broader Train schedule. No formal24pass candidate was trained and
no Dev/Test labels or tracking metrics were read.

The comparison executes two steps of the preserved original `run_fit`,
not a second call to the new loss implementation. In an isolated function
namespace, its explicit permutation-device literal is changed from CUDA to
CPU and its input callbacks read the real cached slices. The original source
file and module bindings used by live GPU jobs are unchanged. The source
and adjusted function hashes are recorded in the adjacent JSON.

The new input path reads actual November IR/RGB, reconstructs all500target
rows and structural descriptors/validity. For this CPU diagnostic, its
encoder callback replays existing B1FPN cache slices. These8frames are within
the earlier12frame actualGPU5B1encoding check, but this CPU run does not
recompute liveGPU features. All referenced RGB/IR FPN and structure arrays
were rehashed; they were not copied.

The following matched exactly:

- Frozen head parameters and buffers, and all9initial adapter tensors.
- CPU RNG after constructing the additional frozen encoder. Initialization
  uses the original2threads; the backbone allocation is isolated with
  `torch.random.fork_rng`.
- Features, target rows, structural descriptors and validity in both batches.
- Original loss components and clipped gradient norm, averaged over the two
  steps; all9updated adapter tensors; all AdamW moments, steps and groups.
- Exact initial RGB-head fallback and normal no-grad feature tensors usable
  by a trainable adapter. No CUDA context was initialized.

Each path performs2diagnostic optimizer steps at seed0. The average loss
and gradient record is `[1.2027819752693176,1.2027819752693176,0.0,
4.7571346167840645e-11,0.01858720276504755]` in the original order
total/supervised/reliable/wrong/gradient. This is an engineering check,
not an accuracy metric or a seed-robustness experiment.

The12frame GPU input check and this CPU optimizer check are separate pieces
of evidence. Full24pass streaming execution, GPU optimizer parity and a
registered matched coverage experiment remain pending. More unique frames
are not assumed to improve HOTA, and training coverage is not counted as a
multimodal innovation. The current S/C/R component family continues on its
unchanged1152cached frames and is unaffected by this preparation.
