# User Guide

## Getting Started

Welcome to the Multi-sensor LMB Filters toolkit! This guide will help you get up and running with multi-object tracking using our advanced filter implementations.

## Installation

### Prerequisites

Before using this toolkit, ensure you have:

- **MATLAB R2022a or later**
- **Statistics and Machine Learning Toolbox**
- **Optimization Toolbox** (recommended)
- At least **4GB RAM** for typical scenarios
- **8GB+ RAM** recommended for large multi-sensor scenarios

### Setup Instructions

1. **Download the toolkit:**
   ```bash
   git clone https://github.com/sachin-deshik-10/MULTISENSOR-LMB-FILTERS.git
   cd MULTISENSOR-LMB-FILTERS
   ```

2. **Initialize MATLAB environment:**
   ```matlab
   % Start MATLAB and navigate to the project directory
   cd('path/to/MULTISENSOR-LMB-FILTERS')
   
   % Add all paths
   setPath;
   
   % Verify installation
   help generateModel
   ```

3. **Test basic functionality:**
   ```matlab
   % Quick test run
   model = generateModel(5, 0.9, 'LBP', 'Fixed');
   fprintf('Installation successful!\n');
   ```

## Basic Usage

### Single-Sensor Tracking

#### Step 1: Configure the Model

```matlab
% Basic configuration
clutterRate = 10;              % Expected clutter per time step
detectionProbability = 0.95;   % Probability of detecting targets
dataAssociation = 'LBP';       % Data association algorithm
scenarioType = 'Fixed';        % Fixed birth locations

model = generateModel(clutterRate, detectionProbability, ...
                     dataAssociation, scenarioType);
```

#### Step 2: Generate or Load Data

```matlab
% Option A: Generate synthetic data
[groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model);

% Option B: Load your own data (see Data Format section)
% measurements = loadYourData('path/to/data.mat');
```

#### Step 3: Run the Filter

```matlab
% Choose filter type
useLmbFilter = true;  % true for LMB, false for LMBM

if useLmbFilter
    stateEstimates = runLmbFilter(model, measurements);
else
    stateEstimates = runLmbmFilter(model, measurements);
end
```

#### Step 4: Visualize Results

```matlab
% Plot tracking results
plotResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs);

% Compute performance metrics
ospaScores = computeSimulationOspa(groundTruth, stateEstimates, model);
fprintf('Average OSPA distance: %.3f\n', mean(ospaScores));
```

### Multi-Sensor Tracking

#### Step 1: Configure Multi-Sensor Model

```matlab
% Multi-sensor configuration
numberOfSensors = 3;
clutterRates = [5, 8, 12];                    % Per-sensor clutter
detectionProbabilities = [0.9, 0.85, 0.95];  % Per-sensor detection
measurementPrecision = [4, 3, 2];            % Measurement quality

model = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, measurementPrecision, 'PU', 'LBP', 'Fixed');
```

#### Step 2: Generate Multi-Sensor Data

```matlab
[groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
```

#### Step 3: Choose Fusion Strategy

```matlab
% Filter options: 'IC', 'PU', 'LMBM'
filterType = 'PU';  % Parallel Update (recommended)

switch filterType
    case 'IC'
        % Iterated-corrector LMB filter
        stateEstimates = runIcLmbFilter(model, measurements);
    case 'PU'
        % Parallel update filters (PU/GA/AA-LMB)
        stateEstimates = runParallelUpdateLmbFilter(model, measurements);
    case 'LMBM'
        % Multi-sensor LMBM filter
        stateEstimates = runMultisensorLmbmFilter(model, measurements);
end
```

#### Step 4: Analyze Results

```matlab
% Visualize multi-sensor results
plotMultisensorResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs);
```

## Advanced Configuration

### Data Association Algorithm Selection

#### Loopy Belief Propagation (Recommended)

```matlab
model.dataAssociationMethod = 'LBP';
model.lbpConvergenceTolerance = 1e-6;
model.maximumNumberOfLbpIterations = 1000;
```

**Best for:** Real-time applications, moderate complexity scenarios
**Pros:** Fast, accurate, good convergence properties
**Cons:** May struggle with highly ambiguous scenarios

#### Gibbs Sampling

```matlab
model.dataAssociationMethod = 'Gibbs';
model.numberOfSamples = 5000;           % MCMC samples
model.gibbsBurnIn = 1000;               % Burn-in samples
```

**Best for:** High accuracy requirements, uncertainty quantification
**Pros:** Asymptotically exact, good for complex scenarios
**Cons:** Slower than LBP, requires convergence monitoring

#### Murty's Algorithm

```matlab
model.dataAssociationMethod = 'Murty';
model.numberOfAssignments = 100;        % K-best assignments
```

**Best for:** Benchmarking, small-scale problems
**Pros:** Exact within K-best, deterministic
**Cons:** Expensive for large problems, may miss probability mass

### Multi-Sensor Fusion Modes

#### Parallel Update (PU-LMB)

```matlab
model.lmbParallelUpdateMode = 'PU';
```

- **Best performance** for independent sensors
- Mathematically rigorous fusion
- Highly parallelizable

#### Geometric Average (GA-LMB)

```matlab
model.lmbParallelUpdateMode = 'GA';
model.sensorWeights = [0.4, 0.3, 0.3];  % Optional weights
```

- **Best localization** accuracy
- Robust to sensor dependencies
- Poor cardinality estimation

#### Arithmetic Average (AA-LMB)

```matlab
model.lmbParallelUpdateMode = 'AA';
model.sensorWeights = [0.4, 0.3, 0.3];  % Optional weights
```

- **Better cardinality** estimation than GA
- Maintains full Gaussian mixtures
- Higher computational cost

### Performance Tuning

#### Memory Management

```matlab
% Track management parameters
model.existenceThreshold = 0.1;         % Prune low-probability tracks
model.minimumTrajectoryLength = 5;      % Keep long trajectories
model.maximumNumberOfObjects = 100;     % Hard limit on objects

% Gaussian mixture management
model.weightThreshold = 1e-4;           % Prune low-weight components
model.maximumComponents = 10;           % Limit mixture complexity
```

#### Computational Efficiency

```matlab
% LBP optimization
model.lbpConvergenceTolerance = 1e-4;   % Relax for speed
model.maximumNumberOfLbpIterations = 500; % Limit iterations

% Gibbs optimization  
model.numberOfSamples = 2000;           % Reduce for speed
model.parallelChains = 4;               % Use multiple chains

% Murty optimization
model.numberOfAssignments = 50;         % Reduce K-best size
```

## Data Formats

### Measurement Data Structure

```matlab
% Single-sensor measurements (cell array over time)
measurements{t} = [x1, y1; x2, y2; ...];  % Measurements at time t

% Multi-sensor measurements (cell array: sensors x time)
measurements{sensor, time} = [x1, y1; x2, y2; ...];
```

### Ground Truth Format

```matlab
groundTruth.labels{t} = [birth_time1, birth_location1; 
                        birth_time2, birth_location2; ...];
groundTruth.states{t} = {[x1; y1; vx1; vy1], [x2; y2; vx2; vy2], ...};
```

### State Estimates Output

```matlab
stateEstimates.labels{t}    % Object labels [2 x N]
stateEstimates.mu{t}        % State estimates {1 x N}  
stateEstimates.Sigma{t}     % Covariance matrices {1 x N}
stateEstimates.objects      % Complete trajectory information
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors

**Problem:** "Out of memory" errors during execution

**Solutions:**
```matlab
% Reduce problem size
model.maximumNumberOfObjects = 50;
model.existenceThreshold = 0.2;

% Use simpler algorithms
model.dataAssociationMethod = 'LBP';  % Instead of Gibbs/Murty
model.lmbParallelUpdateMode = 'PU';   % Instead of AA

% Limit mixture complexity
model.maximumComponents = 5;
```

#### 2. Poor Tracking Performance

**Problem:** High OSPA distances, missed detections

**Solutions:**
```matlab
% Adjust model parameters
model.detectionProbability = 0.95;    % Increase if too low
model.clutterRate = 5;                % Reduce if too high

% Improve data association
model.dataAssociationMethod = 'Gibbs';
model.numberOfSamples = 10000;

% Tune process/measurement noise
model.sigmaQ = 1.0;                   % Process noise
model.sigmaR = 0.5;                   % Measurement noise
```

#### 3. Convergence Issues

**Problem:** LBP fails to converge, Gibbs poor mixing

**Solutions:**
```matlab
% LBP convergence
model.lbpConvergenceTolerance = 1e-4; % Relax tolerance
model.maximumNumberOfLbpIterations = 2000; % Increase iterations

% Gibbs sampling
model.numberOfSamples = 20000;        % More samples
model.gibbsBurnIn = 5000;            % Longer burn-in
model.adaptiveProposal = true;        % Use adaptive proposals
```

### Performance Optimization Tips

#### 1. Algorithm Selection Guide

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|---------|
| Real-time | LBP | Fast convergence |
| High accuracy | Gibbs | Asymptotically exact |
| Benchmarking | Murty | Deterministic, exact |
| Many sensors | PU-LMB | Parallelizable |
| Correlated sensors | GA-LMB | Robust to correlations |
| Cardinality focus | AA-LMB | Better cardinality |

#### 2. Parameter Tuning Workflow

```matlab
% Step 1: Start with conservative settings
model = generateModel(5, 0.9, 'LBP', 'Fixed');

% Step 2: Run baseline
tic; estimates = runLmbFilter(model, measurements); baseline_time = toc;
baseline_ospa = mean(computeSimulationOspa(groundTruth, estimates, model));

% Step 3: Optimize parameters
clutterRates = [5, 10, 15, 20];
detectionProbs = [0.8, 0.9, 0.95, 0.99];

% Grid search or systematic optimization
for i = 1:length(clutterRates)
    for j = 1:length(detectionProbs)
        % Test configuration
        % Record performance
    end
end
```

## Examples Repository

### Example 1: Airport Surveillance

```matlab
% High-clutter environment with multiple sensors
model = generateMultisensorModel(4, [20, 25, 15, 30], ...
    [0.85, 0.90, 0.80, 0.95], [3, 4, 2, 5], 'PU', 'LBP', 'Random', 20);

% Generate complex scenario
[groundTruth, measurements] = generateMultisensorGroundTruth(model, 20);

% Run robust tracking
stateEstimates = runParallelUpdateLmbFilter(model, measurements);

% Analyze performance
ospaScores = computeSimulationOspa(groundTruth, stateEstimates, model);
fprintf('Complex scenario OSPA: %.3f ± %.3f\n', mean(ospaScores), std(ospaScores));
```

### Example 2: Maritime Tracking

```matlab
% Low detection probability, moderate clutter
model = generateMultisensorModel(2, [8, 12], [0.75, 0.80], [6, 4], ...
    'GA', 'Gibbs', 'Fixed');

% Long-duration tracking
model.simulationLength = 200;
[groundTruth, measurements] = generateMultisensorGroundTruth(model);

% Use uncertainty quantification
model.numberOfSamples = 10000;
stateEstimates = runParallelUpdateLmbFilter(model, measurements);

% Analyze track continuity
trackLengths = arrayfun(@(x) x.trajectoryLength, stateEstimates.objects);
fprintf('Average track length: %.1f steps\n', mean(trackLengths));
```

### Example 3: Performance Comparison

```matlab
% Compare all single-sensor algorithms
algorithms = {'LBP', 'Gibbs', 'Murty'};
results = cell(1, 3);
runtimes = zeros(1, 3);
ospaScores = cell(1, 3);

model = generateModel(10, 0.9, 'LBP', 'Fixed');
[groundTruth, measurements] = generateGroundTruth(model);

for i = 1:length(algorithms)
    model.dataAssociationMethod = algorithms{i};
    if strcmp(algorithms{i}, 'Gibbs')
        model.numberOfSamples = 5000;
    elseif strcmp(algorithms{i}, 'Murty')
        model.numberOfAssignments = 100;
    end
    
    tic;
    results{i} = runLmbFilter(model, measurements);
    runtimes(i) = toc;
    
    ospaScores{i} = computeSimulationOspa(groundTruth, results{i}, model);
end

% Display comparison
fprintf('Algorithm Comparison:\n');
for i = 1:length(algorithms)
    fprintf('%s: OSPA=%.3f, Runtime=%.2fs\n', ...
        algorithms{i}, mean(ospaScores{i}), runtimes(i));
end
```

## Further Reading

- **API Reference:** Complete function documentation
- **Algorithm Guide:** Detailed algorithm explanations  
- **Performance Analysis:** Benchmarking and optimization
- **Research Papers:** Theoretical foundations and recent advances

For additional support, please check the GitHub issues page or contact the development team.
