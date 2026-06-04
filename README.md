# Communication-Aware Adaptive KLA

MATLAB research code for communication-aware adaptive Kullback-Leibler average
(KLA) fusion in distributed multi-sensor labelled multi-Bernoulli (LMB)
tracking.

The repository focuses on adaptive GA-LMB and AA-LMB fusion when sensing quality
and communication reliability vary across sensors. It keeps the runnable
experiments, core filter code, and developer documentation, while generated
reports, figures, paper material, and intermediate experiment outputs are kept
out of version control.

## What This Code Implements

- Distributed multi-sensor LMB tracking with local neighborhood fusion.
- Communication-aware measurement delivery, packet-drop, outage, and link
  quality modeling.
- Adaptive KLA weights for GA-LMB and compatible adaptive weights for AA-LMB.
- Branch-decoupled spatial and existence weighting for KLA-style fusion.
- Structure-aware priors derived from local topology and posterior consistency.
- Baseline fusion modes for fixed weights, PD-weighted GA, FI/FID-FIA-inspired
  weighting, PU-LMB, GA-LMB, AA-LMB, IC-LMB, single-sensor LMB, and LMBM.
- Experiment scripts for 4+4 sensor formations, tiered-link ablations, ideal
  communication comparisons, NIS-related diagnostics, and AA validation.

The main adaptive path is:

```text
RUN/GA experiment scripts
  -> multisensorLmb/runDistributedLmbFilter.m
  -> multisensorLmb/runParallelUpdateLmbFilter.m
  -> multisensorLmb/computeAdaptiveFusionWeights.m
  -> multisensorLmb/gaLmbTrackMerging.m
     or multisensorLmb/aaLmbTrackMerging.m
```

## Requirements

- MATLAB R2022a or newer is recommended.
- The code is written for MATLAB-style execution. Some path and unit-level
  checks may run in GNU Octave, but full experiments should be run in MATLAB.
- No external dataset is required. The experiments generate synthetic
  trajectories, measurements, clutter, sensor trajectories, and communication
  events.

The repository includes MEX binaries for the assignment solver on common
platforms. If your platform cannot load the included binaries, use the MATLAB
fallback paths or rebuild `common/assignmentoptimal.c`.

## Quick Start

Clone and enter the repository:

```bash
git clone git@github.com:Deeeeex/Communication-Aware-Adaptive-KLA.git
cd Communication-Aware-Adaptive-KLA
```

Start MATLAB from the repository root and initialize paths:

```matlab
setPath;
```

Run a small single-sensor sanity check:

```matlab
runFilters;
```

Run the compact multi-sensor demo:

```matlab
runMultisensorFilters;
```

Run the main 4+4 formation demo:

```matlab
runMultisensorFilters_formation;
```

## Adaptive KLA Experiment Example

The main GA-LMB adaptive ablation lives in `RUN/GA`. `setPath.m` intentionally
adds only source directories, so add the experiment folder when calling these
scripts directly:

```matlab
setPath;
addpath(fullfile(pwd, 'RUN', 'GA'));

[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    1, ...              % numberOfTrials
    1, ...              % baseSeed
    true, ...           % useFixedSeed
    struct(), ...       % commConfigOverrides
    false, ...          % writeReport
    'fidFiaExistenceRefinement');
```

Use `writeReport = false` for quick development runs. If you set it to `true`,
the generated Markdown and result files are ignored by `.gitignore`.

The recommended adaptive weighting family is configured through
`model.adaptiveFusion`, with the main factors:

```matlab
model.adaptiveFusion.enabled = true;
model.adaptiveFusion.useCovariance = true;
model.adaptiveFusion.useLinkQuality = true;
model.adaptiveFusion.useExistenceConfidence = true;
model.adaptiveFusion.useDecoupledKla = true;
model.adaptiveFusion.useStructureAwareKla = true;
```

The detailed switch definitions and expected behavior are documented in
`docs/ADAPTIVE_FUSION_WEIGHTS_CN.md` and implemented in
`multisensorLmb/computeAdaptiveFusionWeights.m`.

## Useful Experiment Entrypoints

```text
RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m
    Main GA-LMB tiered-link adaptive KLA ablation.

RUN/GA/runMultisensorFilters_formation_4plus4_IdealCommCompare.m
    Ideal-communication comparison for adaptive and fixed fusion modes.

RUN/GA/runMultisensorFilters_formation_4plus4_CommLevelThreeMethodCompare.m
    Communication-level comparison across multiple methods.

RUN/GA/runMultisensorFilters_formation_4plus4_NISCompare.m
    NIS-related diagnostic comparison.

RUN/AA/runAaBalancedCardinalityValidation.m
    AA-LMB validation for Balanced and Cardinality-critical adaptive modes.

RUN/IDEAL/runStandardFixedIdealDistributedCompare.m
    Standard fixed ideal distributed comparison.
```

Most experiment scripts accept a trial count, seed, fixed-seed flag, override
structures, and a report-output flag. Check the function header of each script
for the exact signature.

## Repository Layout

```text
.
├── common/             Shared models, truth generation, communication models,
│                       metrics, association helpers, and plotting utilities.
├── lmb/                Single-sensor LMB prediction, update, association, and
│                       state extraction.
├── lmbm/               Single-sensor LMBM reference implementation.
├── multisensorLmb/     PU, IC, GA, AA, distributed LMB, adaptive KLA weights,
│                       and track merging.
├── multisensorLmbm/    Multi-sensor LMBM reference implementation.
├── RUN/                Reproducible experiment entry scripts.
├── trials/             Runtime and sensitivity trial scripts.
├── docs/               Core code, adaptive weighting, communication, and API
│                       documentation.
└── setPath.m           MATLAB path initializer.
```

Generated experiment files such as `.mat`, `.csv`, `.gif`, `.png`, `.pdf`,
`.log`, `.pid`, report Markdown files under `RUN`, and output directories are
ignored by default.

## Core Files To Read First

- `docs/CORE_CODE_GUIDE_CN.md`: high-level call-chain guide.
- `docs/DYNAMIC_WEIGHT_CORE_CODE_GUIDE_CN.md`: adaptive-weight implementation
  guide.
- `docs/ADAPTIVE_FUSION_WEIGHTS_CN.md`: weighting factors and recommended
  configurations.
- `common/applyCommunicationModel.m`: packet delivery, outage, and link quality.
- `common/evaluateSensorQuality.m`: geometry and FOV-dependent sensing quality.
- `multisensorLmb/runDistributedLmbFilter.m`: local-neighborhood distributed
  wrapper.
- `multisensorLmb/runParallelUpdateLmbFilter.m`: multi-sensor LMB update loop.
- `multisensorLmb/computeAdaptiveFusionWeights.m`: central adaptive weight
  computation.
- `multisensorLmb/gaLmbTrackMerging.m`: GA/KLA fusion consumer.
- `multisensorLmb/aaLmbTrackMerging.m`: AA fusion consumer.

## Lightweight Checks

Path and core-function availability:

```matlab
setPath;
assert(exist('runDistributedLmbFilter', 'file') == 2);
assert(exist('computeAdaptiveFusionWeights', 'file') == 2);
assert(exist('runParallelUpdateLmbFilter', 'file') == 2);
```

Structure-aware adaptive weight regression:

```matlab
test_structure_aware_consistency;
```

Other smoke checks:

```matlab
test_aa_lmb_track_merging;
test_ideal_comm_compare;
test_standard_ideal_distributed_compare;
test_standard_ideal_fixed_compare;
test_state_dependent_quality_false_targets_compare;
```

Some checks run short simulations and may generate ignored output files.

## Notes For Development

- Keep reusable filter behavior in `common/`, `lmb/`, `lmbm/`,
  `multisensorLmb/`, and `multisensorLmbm/`.
- Keep scenario-specific parameter sweeps in `RUN/*/*.m`.
- Do not commit generated reports or experiment outputs.
- When modifying adaptive KLA behavior, start from
  `multisensorLmb/computeAdaptiveFusionWeights.m`, then verify how the
  resulting spatial and existence weights are consumed by
  `gaLmbTrackMerging.m` or `aaLmbTrackMerging.m`.

## License

This project is released under the MIT License. See `LICENSE`.
