# Authors and provenance

`warpmpm` (this package) is a robot-manipulation MPM engine: a typed solver wrapper, a
composable material factory, a force-feedback coupling backend, a MuJoCo adapter, and the
vendored Warp MLS-MPM kernels it drives (`src/warpmpm/kernels/`). The numerical core began as
an external academic project and was extended by our group. This file records who wrote what
and what to cite.

## Vendored numerical core (`src/warpmpm/kernels/`)

The explicit MLS-MPM solver, the quadratic B-spline transfer, and the base materials
(ids 0-8: jelly, metal, sand, foam, snow, plasticine, fluid, stationary, rigid) come from
the UCLA **warp-mpm** of Zeshun Zong and collaborators (Chenfanfu Jiang's group). It is used
in their published work and must be cited when this engine is used.

```bibtex
@inproceedings{zong2023neural,
  author    = {Zong, Zeshun and Li, Xuan and Li, Minchen and Chiaramonte, Maurizio M.
               and Matusik, Wojciech and Grinspun, Eitan and Carlberg, Kevin
               and Jiang, Chenfanfu and Chen, Peter Yichen},
  title     = {Neural Stress Fields for Reduced-Order Elastoplasticity and Fracture},
  booktitle = {SIGGRAPH Asia 2023 Conference Papers},
  doi       = {10.1145/3610548.3618207},
  year      = {2023}
}

@article{xie2023physgaussian,
  author  = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao
             and Yang, Yin and Jiang, Chenfanfu},
  title   = {PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics},
  journal = {arXiv},
  year    = {2023}
}
```

Original upstream contributors (from the fork's git history):
- Zeshun Zong and the MultiPLES group (UCLA): the original solver and base materials.
- supertan0204: the weakly-compressible fluid material.

License boundary:

- Upstream warp-mpm: no license file; its README asks for citation.
- LICENSE (MIT) covers the group's code: everything outside `kernels/`, plus the group
  extensions listed below.
- Upstream-derived parts of `kernels/`: their authors' copyright.

## Group extensions (UT Austin) on top of that core

Layered onto the upstream solver by our group (from git authorship):
- **Cheng-Hsi Hsiao** (`chhsiao@utexas.edu`): moving robot colliders with a velocity boundary,
  multi-material scenes, rigid-body and restitution boundaries, a point-cloud loader, and the
  Warp 1.x compatibility fixes.
- **Krishna Kumar** (`krishnak@utexas.edu`): the Newton-exact grid-impulse contact-force
  (wrist-FT) readout on the velocity collider.
- **Stepan Tretiakov** (`sdt2344@my.utexas.edu`): CUDA support and auto device resolution
  across the entry points, the Warp 1.14 B-spline transpose fix, and the pressure
  covariance sweep experiment.
- The added constitutive models (ids 9-13: `mu_i_sand`, `newtonian`/Bingham/Herschel-Bulkley,
  `mu_i_phi`, `tabulated_viscous`, `tabulated_mu_i`), the local mu(I) / TrackEUCLID granular
  return mappings, and the dough fluid.

Everything outside `kernels/` (the `Solver` wrapper, `Material` factory, `coupling/` backend,
`adapters/` MuJoCo, scenes, examples, tests) is group-authored.

## Borrowed transfer-kernel design: claymore (MIT)

The Step 5 transfer-pipeline optimizations (the fused G2P2G particle pass, per-block
particle binning/sorting, and the shared-memory arena design earmarked for the CUDA
follow-up) port the architecture of [ClaymoreUW](github.com/JustinBonus/claymore).

```bibtex
@article{wang2020massively,
  author  = {Wang, Xinlei and Qiu, Yuxing and Slattery, Stuart R. and Fang, Yu
             and Li, Minchen and Zhu, Song-Chun and Zhu, Yixin and Tang, Min
             and Manocha, Dinesh and Jiang, Chenfanfu},
  title   = {A Massively Parallel and Scalable Multi-GPU Material Point Method},
  journal = {ACM Transactions on Graphics},
  volume  = {39}, number = {4}, year = {2020},
  doi     = {10.1145/3386569.3392442}
}

@phdthesis{bonus2023claymoreuw,
  author = {Bonus, Justin},
  title  = {Evaluation of Fluid-Driven Debris Impacts in a High-Performance
            Multi-GPU Material Point Method},
  school = {University of Washington},
  year   = {2023}
}
```

## Other reference codebases and inspirations

No code copied from any of these.

- GPUMPM (github.com/kuiwuchn/GPUMPM), Gao et al., SIGGRAPH Asia 2018. GPLv3: design
  only. Two-level sparse-block layout of the active-block compute.
- PhysGaussian (github.com/XPandora/PhysGaussian), Xie et al., BibTeX above. No license
  file: design only. The covariance hooks in `kernels/` serve this coupling.
- GeoWarp. Reference for the planned implicit quasi-static solver. License unchecked.
- Genesis (github.com/Genesis-Embodied-AI/Genesis), Apache-2.0. Scene geometry, poses,
  and robot trajectory of the pour example.
- NVIDIA Warp, Apache-2.0. Dependency.
- MuJoCo, Apache-2.0. Dependency.

```bibtex
@article{gao2018gpu,
  author  = {Gao, Ming and Wang, Xinlei and Wu, Kui and Pradhana, Andre
             and Sifakis, Eftychios and Yuksel, Cem and Jiang, Chenfanfu},
  title   = {GPU Optimization of Material Point Methods},
  journal = {ACM Transactions on Graphics},
  volume  = {37}, number = {6}, year = {2018},
  doi     = {10.1145/3272127.3275044}
}
```
