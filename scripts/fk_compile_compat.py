"""Opt-in fix for int/pybind-enum comparison tracing in local torch.compile.

Only the FK forward branch predicate changes. No vendor files are edited and
running processes are unaffected. Call before constructing any compiled step.
Training checkpoints using this helper must record its metadata explicitly.
"""
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap


def install():
    import supermimic.kinematics.g1_fk as module
    cls = module.G1Kinematics
    existing = getattr(cls, '_compile_compat_metadata', None)
    if existing is not None:
        return existing
    original = cls.forward
    source = textwrap.dedent(inspect.getsource(original))
    old = 'l.joint_type == mujoco.mjtJoint.mjJNT_HINGE'
    new = 'l.joint_type == int(mujoco.mjtJoint.mjJNT_HINGE)'
    if source.count(old) != 1:
        raise RuntimeError('FK forward differs from the audited implementation')
    patched = source.replace(old, new)
    filename = '<supermimic_fk_integer_enum_compat>'
    linecache.cache[filename] = (len(patched), None, patched.splitlines(True), filename)
    namespace = dict(module.__dict__)
    exec(compile(patched, filename, 'exec'), namespace)
    fixed = namespace['forward']
    fixed.__qualname__ = original.__qualname__
    cls.forward = fixed
    cls.__call__ = fixed
    metadata = dict(profile='fk_integer_enum_v1', source_file=str(Path(module.__file__).resolve()),
        source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_forward_sha256=hashlib.sha256(source.encode()).hexdigest(),
        corrected_forward_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        change=f'{old} -> {new}',
        reason='Local torch.compile traces integer == MuJoCo pybind enum as false, dropping hinge FK branches.')
    cls._compile_compat_metadata = metadata
    return metadata
