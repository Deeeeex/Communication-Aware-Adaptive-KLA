# API Reference Guide

## Core Filter Functions

### Single-Sensor Filters

#### `runLmbFilter(model, measurements)`

**Description**: Executes the Labelled Multi-Bernoulli (LMB) filter for single-sensor tracking.

**Parameters**:
- `model` (struct): Configuration structure containing filter parameters
- `measurements` (cell array): Time-series measurement data

**Returns**:
- `stateEstimates` (struct): Contains labels, means, covariances, and object trajectories

**Example**:
```matlab
model = generateModel(10, 0.95, 'LBP', 'Fixed');
[~, measurements] = generateGroundTruth(model);
estimates = runLmbFilter(model, measurements);
```

#### `runLmbmFilter(model, measurements)`

**Description**: Executes the LMB Mixture (LMBM) filter with hypothesis management.

**Parameters**:
- `model` (struct): Configuration structure
- `measurements` (cell array): Measurement data

**Returns**:
- `stateEstimates` (struct): State estimates with hypothesis tracking

### Multi-Sensor Filters

#### `runParallelUpdateLmbFilter(model, measurements)`

**Description**: Executes parallel update multi-sensor LMB filters (PU-LMB, GA-LMB, AA-LMB).

**Parameters**:
- `model` (struct): Multi-sensor model configuration
- `measurements` (cell array): Multi-sensor measurement data [sensors x time]

**Returns**:
- `stateEstimates` (struct): Fused multi-sensor state estimates

**Configuration Options**:
- `model.lmbParallelUpdateMode`: 'PU', 'GA', or 'AA'

#### `runIcLmbFilter(model, measurements)`

**Description**: Executes the Iterated-Corrector LMB filter for sequential sensor processing.

**Parameters**:
- `model` (struct): Multi-sensor model configuration  
- `measurements` (cell array): Multi-sensor measurement data

**Returns**:
- `stateEstimates` (struct): Sequential processing results

## Model Generation Functions

### `generateModel(clutterRate, detectionProbability, dataAssociationMethod, scenarioType, numberOfBirthLocations)`

**Description**: Creates single-sensor model configuration.

**Parameters**:
- `clutterRate` (double): Expected clutter returns per time step
- `detectionProbability` (double): Target detection probability [0,1]
- `dataAssociationMethod` (string): 'LBP', 'Gibbs', 'Murty', 'LBPFixed'
- `scenarioType` (string, optional): 'Fixed' or 'Random' birth locations
- `numberOfBirthLocations` (int, optional): Number of birth locations for 'Random'

**Returns**:
- `model` (struct): Complete model configuration

### `generateMultisensorModel(numberOfSensors, clutterRates, detectionProbabilities, q, fusionMode, dataAssociationMethod, scenarioType)`

**Description**: Creates multi-sensor model configuration.

**Parameters**:
- `numberOfSensors` (int): Number of sensors
- `clutterRates` (array): Clutter rates per sensor
- `detectionProbabilities` (array): Detection probabilities per sensor
- `q` (array): Measurement precision parameters per sensor
- `fusionMode` (string): 'IC', 'PU', 'GA', 'AA'
- `dataAssociationMethod` (string): Data association algorithm
- `scenarioType` (string): Scenario configuration

**Returns**:
- `model` (struct): Multi-sensor model configuration

## Data Association Algorithms

### Loopy Belief Propagation

#### `loopyBeliefPropagation(associationMatrices, tolerance, maxIterations)`

**Description**: Executes Williams et al.'s LBP algorithm.

**Parameters**:
- `associationMatrices` (struct): Precomputed association matrices
- `tolerance` (double): Convergence tolerance
- `maxIterations` (int): Maximum number of iterations

**Returns**:
- `r` (array): Posterior existence probabilities
- `W` (matrix): Marginal association probabilities

### Gibbs Sampling

#### `lmbGibbsSampling(associationMatrices, numberOfSamples)`

**Description**: Gibbs sampling for data association.

**Parameters**:
- `associationMatrices` (struct): Association data structures
- `numberOfSamples` (int): Number of MCMC samples

**Returns**:
- `r` (array): Estimated existence probabilities  
- `W` (matrix): Estimated association probabilities

### Murty's Algorithm

#### `lmbMurtysAlgorithm(associationMatrices, numberOfAssignments)`

**Description**: Murty's algorithm for optimal assignment generation.

**Parameters**:
- `associationMatrices` (struct): Cost matrices
- `numberOfAssignments` (int): Number of best assignments to generate

**Returns**:
- `r` (array): Exact existence probabilities
- `W` (matrix): Exact association probabilities

## Performance Evaluation

### `computeSimulationOspa(groundTruth, stateEstimates, model)`

**Description**: Computes OSPA metrics for entire simulation.

**Parameters**:
- `groundTruth` (struct): True object states and trajectories
- `stateEstimates` (struct): Filter output estimates
- `model` (struct): Model configuration

**Returns**:
- `ospaScores` (array): Time-series OSPA distances

### `ospa(X, Y, c, p)`

**Description**: Computes single-step OSPA distance between two sets.

**Parameters**:
- `X` (matrix): First object set [d x n1]
- `Y` (matrix): Second object set [d x n2]  
- `c` (double): Cut-off parameter
- `p` (double): Order parameter

**Returns**:
- `dist` (double): OSPA distance
- `loc` (double): Localization component
- `card` (double): Cardinality component

## Utility Functions

### Matrix Generation

#### `generateLmbAssociationMatrices(objects, measurements, model)`

**Description**: Generates association matrices for LMB filter.

**Parameters**:
- `objects` (struct array): Current object estimates
- `measurements` (cell array): Current time step measurements
- `model` (struct): Model configuration

**Returns**:
- `associationMatrices` (struct): All required matrices for data association
- `posteriorParameters` (struct): Updated spatial distribution parameters

### Track Management

#### `lmbMapCardinalityEstimate(existenceProbabilities)`

**Description**: MAP cardinality estimation for LMB filters.

**Parameters**:
- `existenceProbabilities` (array): Object existence probabilities

**Returns**:
- `nMap` (int): MAP cardinality estimate
- `mapIndices` (array): Indices of MAP objects

## Advanced Features

### Multi-Sensor Track Fusion

#### `puLmbTrackMerging(measurementUpdatedDistributions, priorObjects, model)`

**Description**: Parallel Update LMB track merging.

**Parameters**:
- `measurementUpdatedDistributions` (cell array): Per-sensor updates
- `priorObjects` (struct array): Prior object estimates  
- `model` (struct): Model configuration

**Returns**:
- `fusedObjects` (struct array): Fused multi-sensor estimates

#### `gaLmbTrackMerging(measurementUpdatedDistributions, model)`

**Description**: Geometric Average LMB track merging.

#### `aaLmbTrackMerging(measurementUpdatedDistributions, model)`

**Description**: Arithmetic Average LMB track merging.

### Hypothesis Management (LMBM)

#### `determinePosteriorHypothesisParameters(assignmentVectors, logLikelihoods, posteriorParameters, priorHypothesis)`

**Description**: Updates hypothesis parameters in LMBM filter.

**Parameters**:
- `assignmentVectors` (matrix): Sampled assignment vectors
- `logLikelihoods` (matrix): Assignment log-likelihoods
- `posteriorParameters` (struct): Spatial distribution parameters
- `priorHypothesis` (struct): Prior hypothesis

**Returns**:
- `posteriorHypotheses` (struct array): Updated hypotheses

## Error Handling

### Common Error Codes

- **Model Validation**: Invalid parameter ranges or missing fields
- **Measurement Format**: Incorrect measurement cell array structure  
- **Memory Limits**: Exceeded memory for large multi-sensor LMBM scenarios
- **Convergence**: LBP algorithm convergence failures

### Debugging Tools

#### `validateModel(model)`

**Description**: Validates model configuration for common errors.

**Parameters**:
- `model` (struct): Model to validate

**Returns**:
- `isValid` (logical): Validation result
- `errorMessages` (cell array): List of validation errors

## Performance Optimization Tips

1. **Memory Management**: Use track gating for large scenarios
2. **Computational Efficiency**: Choose appropriate data association method
3. **Parallel Processing**: Enable parallel measurement updates when available
4. **Parameter Tuning**: Optimize convergence tolerances and sample sizes

## Version Compatibility

- **MATLAB R2022a+**: Full compatibility
- **MEX Files**: Platform-specific compilation required
- **Toolbox Dependencies**: Statistics and Machine Learning Toolbox
