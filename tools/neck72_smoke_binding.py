"""Isolate the original fitters' first20Train updates in a smoke-only directory.

Only output path, step/epoch budget and completion labels change in the private
function. Its actual optimizer-step AST remains exactly the registered fitter.
The original source files and formal fit outputs are never edited or reused.
"""
import ast
import hashlib
import importlib
import inspect
import types

from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_gpu


def step_AST(tree):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'step')


class SmokeBudget(ast.NodeTransformer):
    def __init__(self, kind, physical):
        if kind not in {'clean', 'robust'} or physical not in {'4', '5'}:
            raise ValueError('Registered head kind and physicalGPU4/5 only')
        self.folder = 'native_neck_' + kind + '_query'
        self.output = 'neck72_engineering_fits/gpu' + physical + '/' + self.folder
        self.counts = dict(output=0, steps=0, epoch_range=0, epochs_metadata=0, status=0)

    def visit_Constant(self, node):
        if node.value == self.folder:
            self.counts['output'] += 1
            return ast.copy_location(ast.Constant(self.output), node)
        if node.value == 'COMPLETE_NATIVE_ROBUST_QUERY_FIT_STAGE':
            self.counts['status'] += 1
            return ast.copy_location(ast.Constant('SMOKE_NECK72_20_TRAIN_UPDATES'), node)
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'steps':
            self.counts['steps'] += 1; node.value = ast.copy_location(ast.Constant(20), node.value)
        return self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == 'epoch':
            assert ast.dump(node.iter) == ast.dump(ast.parse('range(start_epoch,a.epochs)', mode='eval').body)
            self.counts['epoch_range'] += 1
            node.iter.args[1] = ast.Constant(1)
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == 'epochs' and ast.dump(node.value) == ast.dump(ast.parse('a.epochs', mode='eval').body):
            self.counts['epochs_metadata'] += 1; node.value = ast.Constant(1)
        return self.generic_visit(node)


def make_smoke(kind, physical):
    module = importlib.import_module('tools.train_native_neck_' + kind + '_query')
    source = inspect.getsource(module.main)
    original = ast.parse(source); before_step = ast.dump(step_AST(original))
    change = SmokeBudget(kind, physical)
    tree = ast.fix_missing_locations(change.visit(ast.parse(source)))
    assert change.counts == dict.fromkeys(change.counts, 1)
    assert ast.dump(step_AST(tree)) == before_step
    namespace = dict(module.main.__globals__)
    captured = []
    def frozen(frontend, seed):
        model, info = module.frozen_RGB_model(frontend, seed)
        captured.append((model, state_sha256(model.state_dict())))
        return model, info
    # Probe role retains the actual predecessor/UUID/resource/version checks;
    # formal role additionally requires this probe's future completion proof.
    namespace['require_gpu'] = lambda work, seed, role='fit': require_gpu(work, seed, role='probe')
    namespace['frozen_RGB_model'] = frozen
    exec(compile(tree, inspect.getsourcefile(module.main), 'exec'), namespace)
    entry = namespace['main']
    proof = dict(original_optimizer_step_AST_exact=True, changes=change.counts,
        private_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        original_main_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        output_family=change.output, formal_output_untouched=True,
        scope='First20updates of original Train-only sampler/optimizer/loss, one smoke epoch. '
              'Completion explicitly says smoke1epoch/20updates, not formal18epochs.')
    return entry, proof, captured
