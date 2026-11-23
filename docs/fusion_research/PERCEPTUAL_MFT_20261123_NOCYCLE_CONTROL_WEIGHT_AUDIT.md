# Completed RGB/gray control weight comparison

Same seed0/data/budget, four snapshots per RGB/gray fit. Descriptors for each cue compare RGB structure with itself in the current branch; null and wrong-frame training branches are not self-pairs. Removing the cycle check also changes those branches. Parameter equality is descriptive evidence about these completed fits; it does not prove operator equality on arbitrary unseen inputs or substitute for full Dev/Test. No checkpoint is changed and no accuracy/GT is read.

| Cue | Epoch | Changed tensors | Maximum parameter difference | All tensors exact |
| --- | ---: | ---: | ---: | --- |
| rgb | 6 | 9 | 0.1137033 | False |
| rgb | 12 | 9 | 0.16403346 | False |
| rgb | 18 | 9 | 0.18149099 | False |
| rgb | 24 | 9 | 0.18766868 | False |
| gray | 6 | 9 | 0.12358183 | False |
| gray | 12 | 9 | 0.17814134 | False |
| gray | 18 | 9 | 0.18913801 | False |
| gray | 24 | 9 | 0.19321795 | False |
