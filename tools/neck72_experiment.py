"""Frozen protocol and deterministic cases for the unfinished neck72 route."""
import copy
import json
from pathlib import Path

from tools.neck72_runtime import read, REGISTRY as WORKER_REGISTRY
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/fusion_research'
STUDY = ROOT / 'configs/perceptual_neck72_study_20261124.json'
REGISTRY = 'neck72_study_registered.json'
LENGTHS = {'BT-001': 3000, 'BT-003': 243, 'MSK-002': 1754,
           'PF-001': 15000, 'SN-001': 684, 'SN-015': 6570}
ALGORITHMS = {'ordered': 'ordered_query', 'consensus': 'consensus_query'}
REMOVALS = {
    'missing': {'disable_native_increment': True},
    'feature_mismatch': {'shuffle_native_cue': True},
    'no_alignment': {'disable_native_alignment': True},
    'no_null': {'query_null': False},
    'no_prebirth': {'prebirth_recovery': False},
    'no_covisible': {'preserve_distinct_objects': False},
    'no_object_files': {'object_files': False},
}


def context():
    path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(path) == '388e636d45840f3da8eae44cde2efc4731eeceadc7c169a39f19bf30488ecc03'
    plan = read(path); protocol = read(ROOT / plan['protocol'])
    assert set(protocol['train']) == set(LENGTHS) and sum(LENGTHS.values()) == 27251
    return plan, protocol, Path(plan['work_root'])


def registered():
    plan, protocol, work = context()
    study, registration = read(STUDY), read(work / REGISTRY)
    assert registration['study_sha256'] == sha256(STUDY)
    assert registration['worker_interfaces_sha256'] == sha256(work / WORKER_REGISTRY)
    assert all(sha256(ROOT / p) == h for p, h in registration['source_hashes'].items())
    assert all(sha256(p) == h for p, h in registration['artifact_hashes'].items())
    assert study['fixed_seed'] == 0 and not study['additional_seed_work']
    return plan, protocol, work, study, registration


def head_id(kind, frontend, cue):
    if kind not in {'clean', 'robust'} or frontend not in {'new', 'rgb'} or cue not in {'rgb', 'ir'}:
        raise ValueError('Unknown matched neck72 condition')
    return f'N72_{kind}_{frontend}_{cue}'


def head_specs(work):
    return [dict(id=head_id(kind, frontend, cue), kind=kind, frontend=frontend, cue=cue, seed=0,
                 fit_plan=f'configs/perceptual_native_neck_{kind}_query_20261123.json',
                 worker=f'tools/train_native_neck_{kind}_query.py',
                 output=str(Path(work) / f'native_neck_{kind}_query' / f'{frontend}_{cue}_s0'))
            for kind in ['clean', 'robust'] for frontend in ['new', 'rgb'] for cue in ['rgb', 'ir']]


def config_for(study, head, epoch, algorithm, intervention='normal'):
    if epoch not in [6, 12, 18, 24, 30, 36] or algorithm not in ALGORITHMS:
        raise ValueError('Only preregistered snapshots/backends')
    if intervention not in {'normal', 'no_consensus'} | set(REMOVALS):
        raise ValueError('Unknown preregistered intervention')
    actual_algorithm = 'ordered' if intervention == 'no_consensus' else algorithm
    blueprint = study['blueprints'][f'{actual_algorithm}_{head["frontend"]}_{head["cue"]}']
    config = copy.deepcopy(blueprint['config'])
    fit_plan = ROOT / head['fit_plan']; fitting = read(fit_plan)
    key = f'{head["id"]}_e{epoch:02d}_{algorithm}_{intervention}'
    config.update(id=key, backend=ALGORITHMS[actual_algorithm], native_seed=0,
        native_feature_family='neck_metric72', native_eval_root='native_neck_seed0_eval_features',
        native_query_checkpoint=str(Path(head['output']) / f'epoch{epoch:02d}.pt'),
        native_query_feature_schema=fitting['feature_schema'], native_query_fitting_plan=head['fit_plan'],
        disable_native_increment=False, shuffle_native_cue=False,
        registration='Frozen neck72 seed0 matched representation/loss/control study. '
                     'Original RGB core, prior, frontend and backend constants preserved. '
                     'Not a new fusion architecture or a Test-based primary choice.')
    config.update(REMOVALS.get(intervention, {}))
    return dict(id=key, head=head['id'], kind=head['kind'], frontend=head['frontend'], cue=head['cue'],
                epoch=epoch, algorithm=algorithm, intervention=intervention, config=config)


def baselines():
    return [dict(id='N72_' + arm + '_ByteTrack', baseline=True, frontend=arm,
                 config=dict(id='N72_' + arm + '_ByteTrack', backend='byte', detection_arm=arm))
            for arm in ['rgb', 'new']]


def config_identity(config):
    return json.dumps({k: v for k, v in config.items() if k not in {'id', 'registration'}},
                      sort_keys=True, separators=(',', ':'))


def install_case(work, case):
    path = Path(work) / 'neck72_case_configs' / (case['id'] + '.json')
    if path.exists():
        assert read(path) == case['config'], 'Preserve immutable instantiated study cases'
    else:
        dump(path, case['config'])
    return path


def select_point(curve):
    if not curve or len({p['epoch'] for p in curve}) != len(curve):
        raise ValueError('Missing or duplicated complete-Dev snapshots')
    return min(curve, key=lambda p: (-p['HOTA'], p['epoch']))


def common_budget_decision(curves, epoch):
    if epoch not in [18, 24, 30, 36] or len(curves) != 16:
        raise ValueError('All16 complete matched backend curves are required')
    expected = list(range(6, epoch + 1, 6))
    gains = {}
    for key, curve in curves.items():
        if sorted(p['epoch'] for p in curve) != expected:
            raise ValueError('No stopping on partial/missing Dev curves')
        recent = max(p['HOTA'] for p in curve)
        earlier = max(p['HOTA'] for p in curve if p['epoch'] <= epoch - 12)
        gains[key] = recent - earlier
    extend = epoch < 36 and any(value > .2 + 1e-10 for value in gains.values())
    return dict(common_epochs=epoch, extend_all8_heads=extend, next_epochs=epoch + 6 if extend else epoch,
                gains_vs_best_through12_epochs_earlier=gains, Test_read=False,
                rule='Any of16 complete matched Dev curves gains>0.2pp over its best through12epochs earlier; cap36.')


def selected_cases(study, curves):
    """Freeze all normal/control/removal cases before evaluating any new Test."""
    heads = {head['id']: head for head in study['heads']}
    cases, roles, by_identity = {}, {}, {}

    def add(role, case):
        identity = config_identity(case['config'])
        if identity in by_identity:
            roles[role] = by_identity[identity]
        else:
            assert case['id'] not in cases
            cases[case['id']] = case; by_identity[identity] = case['id']; roles[role] = case['id']
        return roles[role]

    for head in study['heads']:
        for algorithm in ALGORITHMS:
            choice = select_point(curves[head['id'] + '_' + algorithm])
            for intervention in ['normal', 'missing'] + (['feature_mismatch'] if head['cue'] == 'ir' else []):
                role = head['id'] + '_' + algorithm + '_' + intervention
                add(role, config_for(study, head, choice['epoch'], algorithm, intervention))
    primary = study['primary']
    primary_head = heads[primary['head']]
    algorithm = primary['algorithm']
    epoch = select_point(curves[primary_head['id'] + '_' + algorithm])['epoch']
    roles['primary'] = roles[primary_head['id'] + '_' + algorithm + '_normal']
    for name in ['no_alignment', 'no_null', 'no_prebirth', 'no_covisible', 'no_object_files', 'no_consensus']:
        add('primary_' + name, config_for(study, primary_head, epoch, algorithm, name))
    for kind, cue, role in [('clean', 'ir', 'primary_paired_no_KL'),
                            ('robust', 'rgb', 'primary_paired_extra_RGB')]:
        head = heads[head_id(kind, primary_head['frontend'], cue)]
        add(role, config_for(study, head, epoch, algorithm))
    for case in baselines():
        add(case['id'], case)
    return dict(cases=list(cases.values()), roles=roles,
                primary_snapshot=epoch, original16_complete_Dev_selections={
                    key: select_point(curve) for key, curve in curves.items()},
                Test_accuracy_read_for_selection=False)
