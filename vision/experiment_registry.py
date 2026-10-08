"""Stable experiment names and compatibility with the archived configuration names."""
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
with (PROJECT_ROOT / 'reproduction/experiment_aliases.json').open() as handle:
    EXPERIMENT_ALIASES = json.load(handle)
with (PROJECT_ROOT / 'reproduction/config_key_aliases.json').open() as handle:
    CONFIG_KEY_ALIASES = json.load(handle)


def resolve_experiment_name(name, family='VQ'):
    """Resolve the experiment component, preserving an optional repetition suffix."""
    head, separator, tail = str(name).partition('/')
    canonical = EXPERIMENT_ALIASES[family].get(head, head)
    return canonical + separator + tail


def normalize_config(config, kind='train'):
    """Translate old keys without changing values or mutating the caller's config."""
    aliases = {**CONFIG_KEY_ALIASES['common'], **CONFIG_KEY_ALIASES[kind]}

    def convert(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                canonical = aliases.get(key, key)
                if canonical in result:
                    raise ValueError(f'Duplicate configuration key after renaming: {canonical}')
                result[canonical] = convert(item)
            return result
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return value

    return convert(config)


def load_experiment_config(name, family='VQ', config_name='train_config', exp_root=None):
    root = Path(exp_root) if exp_root is not None else PROJECT_ROOT / family / 'exp'
    canonical = resolve_experiment_name(name, family)
    path = root / canonical / f'{config_name}.py'
    # Custom roots may contain an archived checkout with the original directory names.
    if not path.is_file() and (root / name / f'{config_name}.py').is_file():
        path = root / name / f'{config_name}.py'
    if not path.is_file():
        raise FileNotFoundError(f'Experiment configuration not found: {path}')
    spec = importlib.util.spec_from_file_location(f'elpis_{family}_{canonical}_{config_name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kind = 'align' if family == 'LM_align' else ('other' if config_name == 'other_tasks_config' else 'train')
    return normalize_config(module.CONFIG, kind)
