"""Match saved physical states to the target frames used by the evaluator."""
import numpy as np


def reference_frame_indices(qpos, reference_qpos, recorded_root_error):
    # The command advances on reset in the pinned environment. Verify that
    # convention against the saved metric, rather than silently assuming i == i.
    candidates = []
    for offset in (0, 1):
        indices = np.minimum(np.arange(len(qpos)) + offset, len(reference_qpos) - 1)
        errors = np.linalg.norm(qpos[:, :3] - reference_qpos[indices, :3], axis=1)
        residual = float(np.max(np.abs(errors - recorded_root_error)))
        candidates.append((residual, offset, indices))
    residual, offset, indices = min(candidates, key=lambda x: x[0])
    if residual > 2e-5:
        raise ValueError(f"Cannot match reference timing to recorded root error: {residual:g} m")
    return indices, {"reference_frame_offset": offset, "max_root_error_residual_m": residual}
