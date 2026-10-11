# CMUE experiments

Status: NOT_STARTED. No utility estimator has been trained or evaluated. Architecture, teachers, fixed-candidate target and controls are predeclared in ARCHITECTURE_DESIGN.md; cross-fit videos are in the frozen protocol.

Initial loss: Huber utility regression on frozen out-of-fold teacher CE differences, plus a separately documented sign calibration stage. Track teacher provenance, benefit/harm distribution, acceptance frequency, corrections and harmful overrides. RGB fallback must return identical RGB scores when utility is rejected. Compare same evidence, capacity, teacher and training budget against ordinary gates. Do not report module implementation as validation of H1.
