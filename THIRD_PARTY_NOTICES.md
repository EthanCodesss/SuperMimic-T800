# Third-party sources

This project retains the original MIT license and copyright notice from
CuboiLeo/SuperMimic-Reproduced. Local T800 adaptation, experiment wrappers and
documentation are distributed under that same code license unless stated otherwise.

| Source | Pinned revision | Use / license location |
|---|---|---|
| [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | `ba6105349069bf040f1fdad60873d13138d505a2` | Base algorithm; MIT, root `LICENSE` |
| [mjlab](https://github.com/mujocolab/mjlab) | `8bb2bc9e187797689ec82c78c35019afd2fd0376` | Simulator framework; original license restored with vendor snapshot |
| [EngineAI GMR](https://github.com/engineai-robotics/GMR) | `71e02c0598f6d6c6fe1514bf1d69bc24e82f801e` | Retargeting and robot assets; original notices restored with vendor snapshot |
| [EngineAI RL Lab](https://github.com/engineai-robotics/engineai_rl_lab) | `c53f7ec614e34f1ea514cefb935894082f3d4c2a` | Selected reference configuration; BSD-3-Clause, `vendor/engineai_rl_lab_reference/LICENSE.txt` |
| [EngineAI native SDK](https://github.com/engineai-robotics/engineai_robotics_native_sdk) | `335c60e88772c26c7852d0abd6b3c7439037dd8f` | Selected reference XML; BSD-3-Clause, `vendor/engineai_sdk_reference/LICENSE.txt` |

Detailed source hashes are in `configs/sources.lock.json`. Large vendor snapshots
are downloaded from their original repositories, not embedded into this repository.
A local mjlab metadata adjustment lowers its Torch requirement to 2.11; the source
lock records that change.

LAFAN1 and third-party retargeted motion datasets are not covered by the code's MIT
license. Obtain them from the original providers and follow their own terms. Raw
BVH files, retargeted CSV/NPZ files and replay archives containing reference motion
are excluded from Git and Release assets. Published metrics and rendered figures
are experimental documentation; the demo credits the LAFAN1 source motion.
