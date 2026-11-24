"""Complete-Dev decisions for nine fixed-seed spatial/cue conditions.

No files, Test scores, GPU allocation or snapshot mutation in this module.
"""
import math

VARIANTS = ('spatial', 'quality_spatial', 'quality_spatial_redundancy')
CUES = ('rgb', 'gray', 'ir')
NORMAL = tuple((v, c) for v in VARIANTS for c in CUES)


def valid_dev(row):
    if (row.get('split') != 'dev' or row.get('sequences') != ['BT-005', 'SN-013']
            or row.get('frames') != 4834 or row.get('seed') != 0
            or not isinstance(row.get('epoch'), int) or not 1 <= row['epoch'] <= 12):
        raise ValueError('Only complete, fixed-seed official Dev observations may select a model')
    if any(not math.isfinite(float(row['selection_metrics'][k])) for k in ['HOTA', 'AssA', 'IDF1']):
        raise ValueError('Finite full-precision CSV selection fields required')
    if row.get('selection_precision') != 'original276CSV_COMBINED_full_precision':
        raise ValueError('Rounded headline metrics cannot select the spatial primary')


def complete_curves(rows, stage, variants=VARIANTS):
    if stage not in [6, 9, 12] or not variants or any(v not in VARIANTS for v in variants):
        raise ValueError('Registered stage and nonempty variant set required')
    found = {}
    expected = {(v, c, e) for v in variants for c in CUES for e in range(1, stage+1)}
    for row in rows:
        valid_dev(row)
        key = row['variant'], row['cue'], row['epoch']
        if (key not in expected or key in found or row.get('removal')
                or row.get('intervention', 'normal') != 'normal'):
            raise ValueError('Unique normal Dev curves are required for every variant/cue/epoch')
        found[key] = row
    if set(found) != expected:
        raise ValueError('Missing required complete-Dev snapshot; do not silently drop a negative fit')
    return found


def complete_interventions(rows, stage, variants=VARIANTS):
    expected = {(v, e, i) for v in variants for e in range(1, stage+1)
                for i in ['unavailable', 'IR_image_spatial_roll']}
    found = set()
    for row in rows:
        valid_dev(row)
        key = row['variant'], row['epoch'], row.get('intervention')
        if key not in expected or key in found or row['cue'] != 'ir' or row.get('removal'):
            raise ValueError('Every registered IR intervention needs one complete official Dev observation')
        found.add(key)
    if found != expected:
        raise ValueError('Incomplete actual-image-roll/missing modality coverage')


def best_dev(rows):
    if not rows:
        raise ValueError('Cannot select an empty curve')
    for row in rows:
        valid_dev(row)
        if row.get('intervention', 'normal') != 'normal':
            raise ValueError('Do not select an intervened model as the primary')
    return min(rows, key=lambda r: (-r['selection_metrics']['HOTA'], -r['selection_metrics']['AssA'],
        -r['selection_metrics']['IDF1'], r['epoch'], VARIANTS.index(r['variant'])))


def budget_decision(rows, interventions, stage):
    curves = complete_curves(rows, stage)
    complete_interventions(interventions, stage)
    comparisons = []
    for variant, cue in NORMAL:
        values = [curves[variant, cue, e] for e in range(1, stage+1)]
        before = best_dev([r for r in values if r['epoch'] <= stage-3])
        now = best_dev(values)
        comparisons.append(dict(variant=variant, cue=cue,
            improvement_pp=now['selection_metrics']['HOTA']-before['selection_metrics']['HOTA']))
    extend = stage < 12 and any(r['improvement_pp'] > .2+1e-10 for r in comparisons)
    return dict(completed_stage=stage, extend_all_nine=extend, next_stage=stage+3 if extend else None,
        comparisons=comparisons, selection_input_split='dev', numerical_convergence_claim=False)


def select_models(rows, stage, direct_rgb_dev, fresh_rgb_dev):
    complete_curves(rows, stage)
    if not all(math.isfinite(float(x)) for x in [direct_rgb_dev, fresh_rgb_dev]):
        raise ValueError('Both complete RGB source values are required')
    normals = [best_dev([r for r in rows if (r['variant'], r['cue']) == key]) for key in NORMAL]
    primary = best_dev([r for r in rows if r['cue'] == 'ir'])
    rgb = next(r for r in normals if (r['variant'], r['cue']) == (primary['variant'], 'rgb'))
    gray = next(r for r in normals if (r['variant'], r['cue']) == (primary['variant'], 'gray'))
    threshold = max(float(direct_rgb_dev), float(fresh_rgb_dev),
        rgb['selection_metrics']['HOTA'], gray['selection_metrics']['HOTA'])
    return dict(primary=primary, all_nine_own_Dev_best=normals, primary_variant_RGB=rgb,
        primary_variant_gray=gray, removal_gate_threshold_Dev_HOTA=threshold,
        fit_enabled_removals=primary['selection_metrics']['HOTA'] > threshold,
        all_nine_still_require_complete_Test=True, selection_input_split='dev',
        Test_accuracy_used=False, goal_met=False, supported_innovations=0)
