"""Pure Dev-only F4 design rules; file provenance and executors remain separate.

Tests supply synthetic records, never trained models or observed HOTA. A record's
coverage assertions are not file verification. This module cannot load a model,
launch inference, establish eligibility from files, or authorize Test execution.
"""
import copy
import math
import re

ARCHITECTURES = {'native': ('quality', 'temporal'),
                 'spatial': ('spatial', 'quality_spatial', 'quality_spatial_redundancy')}
PROFILES = ('rgb_uniform', 'rgb_attention_proxy', 'fused_uniform', 'fused_attention_proxy')
PREFERENCE = ('byte', 'rgb_uniform', 'fused_uniform', 'rgb_attention_proxy', 'fused_attention_proxy')
WEIGHTS = (0., .05, .1)
INTERVENTIONS = ('IR_unavailable', 'IR_image_spatial_roll')
DEV_SEQUENCES = ['BT-005', 'SN-013']


def complete_dev(row, intervention='normal'):
    if (row.get('split') != 'dev' or row.get('frames') != 4834 or
            row.get('sequences') != DEV_SEQUENCES or row.get('seed') != 0 or
            row.get('intervention', 'normal') != intervention):
        raise ValueError('Require complete seed0 full Dev of the declared intervention')
    for key in ['HOTA', 'AssA', 'IDF1']:
        value = row['metrics'][key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
            raise ValueError('Finite full-precision official Dev values required')
    return row


def score(row):
    return tuple(-float(row['metrics'][key]) for key in ['HOTA', 'AssA', 'IDF1'])


def visual_best(rows):
    return min(rows, key=lambda r: (*score(r), r['epoch'], r['architecture'], r['id']))


def parent_details(record):
    family = record['family']
    if (family not in ARCHITECTURES or record.get('normal_curves_complete') is not True or
            record.get('original_primary_frozen') is not True or record.get('interventions_complete') is not True):
        raise ValueError('Both completed original visual families and frozen choices are required')
    normal = record['normal_best']
    expected = {(arch, cue) for arch in ARCHITECTURES[family] for cue in ['rgb', 'gray', 'ir']}
    found = {}
    for row in normal:
        complete_dev(row)
        key = row['architecture'], row['cue']
        if key not in expected or key in found or row.get('family') != family:
            raise ValueError('Every matched cue/architecture appears exactly once')
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) or
                not re.fullmatch(r'[0-9a-f]{64}', row['checkpoint_sha256']) or
                not isinstance(row['epoch'], int) or not 1 <= row['epoch'] <= 12):
            raise ValueError('Explicit frozen visual identity, checkpoint and logical epoch required')
        found[key] = row
    if set(found) != expected or len({r['id'] for r in normal}) != len(normal):
        raise ValueError('All matched original normal arms must be complete and unique')
    matches = [r for r in normal if r['id'] == record['primary_id'] and r['cue'] == 'ir']
    if len(matches) != 1:
        raise ValueError('Use the original frozen IR primary, not a new selected normal row')
    primary = matches[0]
    controls = dict(matched_RGB=found[primary['architecture'], 'rgb'],
        matched_gray=found[primary['architecture'], 'gray'],
        global_RGB=visual_best([r for r in normal if r['cue'] == 'rgb']),
        global_gray=visual_best([r for r in normal if r['cue'] == 'gray']))
    for name in ['canonical_RGB', 'fresh_RGB']:
        row = complete_dev(record[name])
        if row['cue'] != 'rgb' or not row.get('baseline'):
            raise ValueError('Explicit canonical/fresh direct RGB baseline records required')
        controls[name] = row
    for intervention in INTERVENTIONS:
        row = complete_dev(record['interventions'][intervention], intervention)
        if (row['cue'] != 'ir' or row['source_primary_id'] != primary['id'] or
                row['checkpoint_sha256'] != primary['checkpoint_sha256'] or
                any(row[key] != primary[key] for key in ['family', 'architecture', 'epoch'])):
            raise ValueError('Interventions must use the identical original IR snapshot')
        controls[intervention] = row
    deltas = {name: primary['metrics']['HOTA']-r['metrics']['HOTA'] for name, r in controls.items()}
    return dict(family=family, primary=primary, controls=controls,
        Dev_HOTA_deltas=deltas, eligible=all(value > 1e-10 for value in deltas.values()))


def select_visual_parent(records):
    if len(records) != 2 or {r['family'] for r in records} != set(ARCHITECTURES):
        raise ValueError('Wait for both original native and spatial Dev families')
    details = [parent_details(r) for r in records]
    eligible = [r for r in details if r['eligible']]
    chosen = min(eligible, key=lambda r: (*score(r['primary']),
        0 if r['family'] == 'native' else 1)) if eligible else None
    return dict(status='ELIGIBLE_IN_SUPPLIED_DEV_RECORDS' if chosen else 'NO_ELIGIBLE_VISUAL_PARENT',
        chosen_family=None if chosen is None else chosen['family'],
        chosen_primary_id=None if chosen is None else chosen['primary']['id'],
        records=copy.deepcopy(records), details=copy.deepcopy(details),
        file_provenance_not_verified_by_pure_rules=True, formal_launch_ready=False,
        four_direction_completion=False, goal_met=False, supported_innovations=0)


def case_id(visual, profile, weight, suffix=''):
    return f"FM_{visual['family']}_{visual['id']}_{profile}_w{round(weight*100):03d}"+suffix


def make_case(visual, profile, weight, *, floor=.2, intervention='normal'):
    if profile not in PREFERENCE or weight not in WEIGHTS or intervention not in ('normal', *INTERVENTIONS):
        raise ValueError('Only registered profiles, weights and interventions')
    if (profile == 'byte') != (weight == 0):
        raise ValueError('Weight0 is one shared ByteTrack case, without feature reads')
    if floor not in [0., .2]:
        raise ValueError('Only reliability-floor removal is registered')
    suffix = '' if floor == .2 else '_no_reliability_floor'
    if intervention != 'normal':
        if visual['cue'] != 'ir':
            raise ValueError('Actual IR interventions only apply to the chosen IR visual parent')
        suffix += '_'+intervention
    return dict(id=case_id(visual, profile, weight, suffix), visual=copy.deepcopy(visual),
        visual_key=visual['family']+':'+visual['id'], profile=profile, appearance_weight=weight,
        feature_source=None if profile == 'byte' else profile.split('_', 1)[0],
        quality_policy='attention_proxy' if profile.endswith('attention_proxy') else 'uniform',
        minimum_reliability=floor, score_acceptance_threshold=.6,
        positive_reliability_and_valid_embedding_still_required=True,
        intervention=intervention, fixed_seed=0)


def normal_case_plan(parent_selection):
    family = parent_selection['chosen_family']
    if family is None:
        raise ValueError('No learned fusion association without an eligible visual parent')
    record = next(r for r in parent_selection['records'] if r['family'] == family)
    details = parent_details(record)
    if not details['eligible'] or details['primary']['id'] != parent_selection['chosen_primary_id']:
        raise ValueError('Parent selection no longer agrees with original supplied Dev')
    roles = dict(primary_IR=details['primary'], **{name: details['controls'][name] for name in
        ['matched_RGB', 'matched_gray', 'global_RGB', 'global_gray', 'fresh_RGB']})
    unique, role_keys = {}, {}
    for role, row in roles.items():
        key = row['family']+':'+row['id']
        if key in unique and unique[key] != row:
            raise ValueError('Visual ID reuse requires identical complete metadata, not equal scores')
        unique[key], role_keys[role] = copy.deepcopy(row), key
    cases = []
    for visual in unique.values():
        cases.append(make_case(visual, 'byte', 0.))
        profiles = ('rgb_uniform',) if visual.get('baseline') else PROFILES
        for profile in profiles:
            for weight in WEIGHTS[1:]:
                cases.append(make_case(visual, profile, weight))
    assert len(cases) <= 48 and len({r['id'] for r in cases}) == len(cases)
    return dict(parent=copy.deepcopy(parent_selection), visuals=unique, role_visual_keys=role_keys,
        normal_cases=cases, all_normal_cases_require_full_Test=True,
        model_fits_added=0, formal_launch_ready=False)


def complete_case_scores(cases, rows):
    wanted = {c['id']: c for c in cases}
    if set(rows) != set(wanted):
        raise ValueError('Every planned case needs complete Dev; do not discard negative cases')
    for identifier, case in wanted.items():
        row = complete_dev(rows[identifier], case['intervention'])
        if row['id'] != identifier:
            raise ValueError('Dev results must identify the exact case')
        if case['appearance_weight'] == 0 and case['intervention'] == 'normal':
            for key in ['HOTA', 'AssA', 'IDF1']:
                if abs(row['metrics'][key]-case['visual']['metrics'][key]) > 1e-10:
                    raise ValueError('Normal weight0 must reproduce its original visual parent metrics')
    return rows


def best_case(cases, rows):
    return min(cases, key=lambda c: (*score(rows[c['id']]), c['appearance_weight'],
        PREFERENCE.index(c['profile']), c['id']))


def profile_cases(plan, visual_key, profile):
    return [r for r in plan['normal_cases'] if r['visual_key'] == visual_key and r['profile'] in ['byte', profile]]


def select_memory(plan, normal_scores):
    complete_case_scores(plan['normal_cases'], normal_scores)
    best = {key: best_case([c for c in plan['normal_cases'] if c['visual_key'] == key], normal_scores)
            for key in plan['visuals']}
    key = plan['role_visual_keys']['primary_IR']
    ir = [c for c in plan['normal_cases'] if c['visual_key'] == key]
    fused = best_case([c for c in ir if c['profile'] == 'byte' or c['feature_source'] == 'fused'], normal_scores)
    uniform = best_case(profile_cases(plan, key, 'rgb_uniform'), normal_scores)
    same_q = best_case(profile_cases(plan, key,
        'rgb_attention_proxy' if fused['quality_policy'] == 'attention_proxy' else 'rgb_uniform'), normal_scores)
    zero = next(c for c in ir if c['profile'] == 'byte')
    def at_weight(reference, profile):
        return next(c for c in profile_cases(plan, key, profile)
                    if c['appearance_weight'] == reference['appearance_weight'])
    primary = best[key]
    primary_q = 'rgb_attention_proxy' if primary['quality_policy'] == 'attention_proxy' else 'rgb_uniform'
    fused_q = 'rgb_attention_proxy' if fused['quality_policy'] == 'attention_proxy' else 'rgb_uniform'
    h = normal_scores[fused['id']]['metrics']['HOTA']
    floor_gate = fused['appearance_weight'] > 0 and all(h > normal_scores[c['id']]['metrics']['HOTA']+1e-10
                                                      for c in [uniform, same_q, zero])
    return dict(primary_IR=primary, best_fused_IR=fused,
        per_visual_best=best, role_best={role: best[key] for role, key in plan['role_visual_keys'].items()},
        own_visual_RGB_uniform_best=uniform, own_visual_same_q_RGB_best=same_q,
        RGB_uniform_at_primary_IR_weight=at_weight(primary, 'rgb_uniform'),
        RGB_same_q_at_primary_IR_weight=at_weight(primary, primary_q),
        RGB_uniform_at_best_fused_weight=at_weight(fused, 'rgb_uniform'),
        RGB_same_q_at_best_fused_weight=at_weight(fused, fused_q),
        fit_reliability_floor_removal=bool(floor_gate), ordinary_RGB_gain_not_fusion_memory=True,
        normal_Dev_only_selection=True, primary_uses_fused_memory=best[key]['feature_source'] == 'fused',
        formal_launch_ready=False, goal_met=False, supported_innovations=0)


def extra_cases(plan, selection):
    primary_key = plan['role_visual_keys']['primary_IR']
    normal_ir = [c for c in plan['normal_cases'] if c['visual_key'] == primary_key]
    result = [make_case(c['visual'], c['profile'], c['appearance_weight'], intervention=intervention)
              for c in normal_ir for intervention in INTERVENTIONS]
    if selection['fit_reliability_floor_removal']:
        chosen = selection['best_fused_IR']
        matched_rgb = 'rgb_attention_proxy' if chosen['quality_policy'] == 'attention_proxy' else 'rgb_uniform'
        profiles = sorted({chosen['profile'], 'rgb_uniform', matched_rgb})
        fresh = plan['visuals'][plan['role_visual_keys']['fresh_RGB']]
        for weight in WEIGHTS[1:]:
            result.extend(make_case(chosen['visual'], profile, weight, floor=0.) for profile in profiles)
            result.extend(make_case(chosen['visual'], chosen['profile'], weight, floor=0., intervention=i)
                          for i in INTERVENTIONS)
            result.append(make_case(fresh, 'rgb_uniform', weight, floor=0.))
    assert len({r['id'] for r in result}) == len(result)
    return result


def freeze_case_plan(plan, normal_scores, additional_scores):
    selection = select_memory(plan, normal_scores)
    additions = extra_cases(plan, selection)
    complete_case_scores(additions, additional_scores)
    scores = {**normal_scores, **additional_scores}
    all_cases = plan['normal_cases']+additions
    fresh_key = plan['role_visual_keys']['fresh_RGB']
    original_parent = next(d for d in plan['parent']['details'] if d['family'] == plan['parent']['chosen_family'])
    for case in additions:
        if case['appearance_weight'] == 0:
            reference = original_parent['controls'][case['intervention']]['metrics']
            if any(abs(scores[case['id']]['metrics'][key]-reference[key]) > 1e-10
                   for key in ['HOTA', 'AssA', 'IDF1']):
                raise ValueError('Intervention weight0 must reproduce its original visual parent result')
        if case['intervention'] != 'IR_unavailable':
            continue
        control = next(c for c in all_cases if c['visual_key'] == fresh_key and
            c['intervention'] == 'normal' and c['appearance_weight'] == case['appearance_weight'] and
            (c['profile'] == 'byte' or (c['profile'] == 'rgb_uniform' and
                                      c['minimum_reliability'] == case['minimum_reliability'])))
        if any(abs(scores[case['id']]['metrics'][key]-scores[control['id']]['metrics'][key]) > 1e-10
               for key in ['HOTA', 'AssA', 'IDF1']):
            raise ValueError('Missing IR must equal fresh RGB q1 at the same weight AND reliability floor')
    # No normal choice depends on wrong/missing or removal outcomes.
    frozen = plan['normal_cases']+additions
    assert len(frozen) <= 78 and len({r['id'] for r in frozen}) == len(frozen)
    removal = [c for c in additions if c['intervention'] == 'normal' and
               c['visual_key'] == selection['best_fused_IR']['visual_key'] and
               c['profile'] == selection['best_fused_IR']['profile']]
    policies = None
    if removal:
        zero = next(c for c in plan['normal_cases'] if c['visual_key'] == selection['best_fused_IR']['visual_key']
                    and c['profile'] == 'byte')
        own = best_case([zero, *removal], scores)
        paired = next(c for c in removal if c['appearance_weight'] == selection['best_fused_IR']['appearance_weight'])
        policies = dict(own_Dev_best=own['id'], paired_best_fused_weight=paired['id'],
                        zero_weight_reused=zero['id'])
    strong_controls = {}
    for role, key in plan['role_visual_keys'].items():
        if role == 'primary_IR':
            continue
        choices = [c for c in all_cases if c['visual_key'] == key and c['intervention'] == 'normal']
        strong_controls[role] = best_case(choices, scores)['id']
    return dict(status='COMPLETE_DEV_ONLY_F4_CASE_PLAN_PENDING_FILE_VERIFICATION',
        selection=selection, normal_cases=len(plan['normal_cases']), additional_cases=len(additions),
        cases=frozen, reliability_floor_removal_policies=policies,
        strong_control_roles_including_registered_floor_removals=strong_controls,
        all_normal_negative_cases_retained=True, all_cases_require_full7Test15985=True,
        Test_accuracy_used=False, file_provenance_not_verified_by_pure_rules=True,
        formal_launch_ready=False, four_direction_completion=False, goal_met=False, supported_innovations=0)
