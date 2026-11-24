"""Independent contrast/quality attribution rules; no I/O or GPU launch.

The four cells keep RGB protection and any causal temporal mechanism fixed.
Positive effects here are necessary numerical evidence, not novelty claims.
"""
import math

from tools.native_pyramid_experiment import make_arm
from tools.native_pyramid_study_rules import best_dev

CELLS = ('00', '10', '01', '11')  # contrast, learned fusion-quality gate
DEV_SEQUENCES = ['BT-005', 'SN-013']


def module_kwargs(study, mode, cell):
    if cell not in CELLS:
        raise ValueError('The complete contrast/quality two-factor design is fixed')
    value = make_arm(study, mode+'_ir', mode, 'ir')['module_kwargs']
    value.update(spectral=cell[0] == '1', use_quality=cell[1] == '1')
    return value


def effects(hota):
    """HOTA inputs and all outputs are absolute percentage points."""
    if set(hota) != set(CELLS) or any(not math.isfinite(float(v)) for v in hota.values()):
        raise ValueError('All four finite HOTA cells are required; missing is not zero')
    if any(not 0 <= float(v) <= 100 for v in hota.values()):
        raise ValueError('HOTA must use the percentage-point scale')
    base, contrast, quality, both = (float(hota[c]) for c in CELLS)
    return dict(contrast_alone=contrast-base, quality_alone=quality-base,
        contrast_given_quality=both-quality, quality_given_contrast=both-contrast,
        combined=both-base, interaction=both-contrast-quality+base)


def select_complete_dev(rows, through_epoch, primary_epoch, mode):
    """Preserve the parent's primary and budget; select no primary from Test."""
    if through_epoch not in [6, 9, 12] or not 1 <= primary_epoch <= through_epoch:
        raise ValueError('Use the already frozen parent budget and primary epoch')
    if mode not in ['quality', 'temporal']:
        raise ValueError('Use the frozen parent primary mode')
    expected = {(cell, epoch) for cell in CELLS for epoch in range(1, through_epoch+1)}
    found = {}
    lineage = set()
    for row in rows:
        key = row['cell'], row['epoch']
        if (key not in expected or key in found or row['split'] != 'dev' or
                row['sequences'] != DEV_SEQUENCES or row['frames'] != 4834 or
                row['seed'] != 0 or row['cue'] != 'ir' or row['mode'] != mode or
                row.get('intervention', 'normal') != 'normal' or
                row['executed_fit_budget'] != through_epoch):
            raise ValueError('Need complete matched, normal seed0 Dev curves for all four cells')
        if any(not math.isfinite(float(row['metrics'][k])) for k in ['HOTA', 'AssA', 'IDF1']):
            raise ValueError('Nonfinite official Dev metrics')
        if row['complete_Train_frames_per_epoch'] != 27251:
            raise ValueError('A sampled Train subset cannot supply a matched factorial cell')
        lineage.add((row['source_RGB_sha256'], row['initial_native_sha256'],
                     row['Train_schedule_sha256']))
        found[key] = row
    if set(found) != expected or len(lineage) != 1:
        raise ValueError('Missing cells/epochs or unequal source, initialization or Train schedule')
    paired = {cell: found[cell, primary_epoch] for cell in CELLS}
    own = {cell: best_dev([found[cell, epoch] for epoch in range(1, through_epoch+1)]) for cell in CELLS}
    return dict(paired_parent_primary_epoch=paired, own_Dev_best=own,
        fixed_parent_primary_mode=mode, fixed_parent_primary_epoch=primary_epoch,
        executed_complete_Train_epochs_each=through_epoch,
        Test_used_for_selection=False, every_selected_cell_requires_full_Test=True)


def necessary_numeric_evidence(dev_hota, test_hota):
    """Evaluate frozen cells without promoting a Test winner or counting novelty."""
    dev, test = effects(dev_hota), effects(test_hota)
    candidates = {}
    for name, single, conditional in [
            ('contrast', 'contrast_alone', 'contrast_given_quality'),
            ('fusion_quality', 'quality_alone', 'quality_given_contrast')]:
        candidates[name] = dict(
            positive_alone_on_Dev_and_Test=all(value[single] > 1e-10 for value in [dev, test]),
            positive_given_other_on_Dev_and_Test=all(value[conditional] > 1e-10 for value in [dev, test]))
        candidates[name]['necessary_four_effects_positive'] = all(candidates[name].values())
    return dict(Dev=dev, Test=test, candidate_mechanism_effects=candidates,
        positive_mechanism_effects=sum(v['necessary_four_effects_positive'] for v in candidates.values()),
        supported_innovations=0, goal_met=False,
        additional_required_evidence=['Matched RGB and gray controls', 'Full official metrics and source coverage',
            'Actual missing/wrong IR controls', 'Prior-art distinction and mechanistic analysis',
            'Full-system measured cost', 'Fixed HOTA target and complete study requirements'])
