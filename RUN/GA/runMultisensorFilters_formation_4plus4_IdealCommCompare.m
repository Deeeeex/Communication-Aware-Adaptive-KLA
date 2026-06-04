function [reportPath, summary] = runMultisensorFilters_formation_4plus4_IdealCommCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, adaptiveFusionOverrides)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_IDEALCOMMCOMPARE
% Compare ordinary GA against structure-aware decoupled KLA under ideal communication.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 5;
end
if nargin < 2 || isempty(baseSeed)
    baseSeed = 1;
end
if nargin < 3 || isempty(useFixedSeed)
    useFixedSeed = true;
end
if nargin < 4 || isempty(writeReport)
    writeReport = true;
end
if nargin < 5 || isempty(adaptiveFusionOverrides)
    adaptiveFusionOverrides = struct();
end

reportPath = '';
summary = struct();

staggeredBirths = true;
leaderSensor = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis';

adaptiveFusionConfig = struct( ...
    'enabled', true, ...
    'emaAlpha', 0.7, ...
    'minWeight', 0.05, ...
    'useCovariance', true, ...
    'useLinkQuality', true, ...
    'useCardinalityConsensus', false, ...
    'useExistenceConfidence', true, ...
    'existenceConfidenceMinScore', 0.85, ...
    'existenceConfidencePower', 2.0, ...
    'useDecoupledKla', true, ...
    'spatialEmaAlpha', 0.7, ...
    'existenceEmaAlpha', 0.7, ...
    'spatialMinWeight', 0.05, ...
    'existenceMinWeight', 0.05, ...
    'spatialCovariancePower', 1.0, ...
    'spatialLinkQualityPower', 1.0, ...
    'existenceLinkQualityPower', 1.0, ...
    'existenceConfidenceWeightPower', 1.0, ...
    'spatialDecouplingStrength', 0.5, ...
    'existenceDecouplingStrength', 0.15, ...
    'useStructureAwareKla', true, ...
    'usePosteriorStructureConsistency', false, ...
    'spatialStructureStrength', 0.45, ...
    'existenceStructureStrength', 0.08, ...
    'structureReliabilityPower', 0.30, ...
    'structureReliabilityMinScore', 0.25, ...
    'useFreshness', false, ...
    'useHistory', false, ...
    'useNIS', false, ...
    'robustNIS', false);
adaptiveFusionConfig = mergeStructFields(adaptiveFusionConfig, adaptiveFusionOverrides);

numberOfSensors = 8;
clutterRates = 3 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);

commConfig = struct();
commConfig.level = 0;
commConfig.globalMaxMeasurementsPerStep = inf;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.0;
commConfig.pDropBySensor = zeros(1, numberOfSensors);

sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.0;
sensorMotionConfig.initialStates = buildSensorInitialStates();

targetFormationConfig = struct();
targetFormationConfig.targetFormationEnabled = true;
targetFormationConfig.targetFormationStaggeredBirths = staggeredBirths;
targetFormationConfig.targetFormationBirthInterval = 8;
targetFormationConfig.targetFormationStartTime = 1;
targetFormationConfig.targetFormationLifeSpan = 100;
targetFormationConfig.targetBirthStates = buildTargetBirthStates();
targetFormationConfig.targetFormationCount = size(targetFormationConfig.targetBirthStates, 2);

eOspaBase = zeros(numberOfTrials, numberOfSensors);
hOspaBase = zeros(numberOfTrials, numberOfSensors);
rmseBase = zeros(numberOfTrials, numberOfSensors);
eOspaAdaptive = zeros(numberOfTrials, numberOfSensors);
hOspaAdaptive = zeros(numberOfTrials, numberOfSensors);
rmseAdaptive = zeros(numberOfTrials, numberOfSensors);
consOspaBase = zeros(numberOfTrials, 1);
consOspaAdaptive = zeros(numberOfTrials, 1);
consPosBase = zeros(numberOfTrials, 1);
consPosAdaptive = zeros(numberOfTrials, 1);
consCardBase = zeros(numberOfTrials, 1);
consCardAdaptive = zeros(numberOfTrials, 1);
pDropBySensorTrials = zeros(numberOfTrials, numberOfSensors);

for trial = 1:numberOfTrials
    fprintf('Trial %d/%d\n', trial, numberOfTrials);
    if useFixedSeed
        rng(baseSeed + trial);
    end

    model = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'GA', 'LBP', 'Formation', ...
        sensorMotionConfig, targetFormationConfig);
    model.sensorCommRange = sensorCommRange;
    model.fusionWeighting = fusionWeighting;
    model.sensorFovEnabled = true;
    model.sensorFovHalfAngleDeg = 60;
    model.sensorFovRange = 60000;

    [~, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
    model.sensorTrajectories = sensorTrajectories;

    [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
    pDropBySensorTrials(trial, :) = reshape(commStats.pDropBySensor, 1, []);

    modelBase = model;
    modelBase.adaptiveFusion = adaptiveFusionConfig;
    modelBase.adaptiveFusion.enabled = false;
    baseNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorBase, localModelsBase] = runDistributedLmbFilter( ...
        modelBase, measurementsDelivered, sensorTrajectories, baseNeighborMap, commStats);

    modelAdaptive = model;
    modelAdaptive.adaptiveFusion = adaptiveFusionConfig;
    modelAdaptive.adaptiveFusion.enabled = true;
    adaptiveNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorAdaptive, localModelsAdaptive] = runDistributedLmbFilter( ...
        modelAdaptive, measurementsDelivered, sensorTrajectories, adaptiveNeighborMap, commStats);

    for s = 1:numberOfSensors
        [eBase, hBase] = computeSimulationOspa(localModelsBase{s}, groundTruthRfs, stateEstimatesBySensorBase{s});
        [eAdaptive, hAdaptive] = computeSimulationOspa(localModelsAdaptive{s}, groundTruthRfs, stateEstimatesBySensorAdaptive{s});
        eOspaBase(trial, s) = mean(eBase);
        hOspaBase(trial, s) = mean(hBase);
        rmseBase(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorBase{s}, groundTruthRfs), 'omitnan');
        eOspaAdaptive(trial, s) = mean(eAdaptive);
        hOspaAdaptive(trial, s) = mean(hAdaptive);
        rmseAdaptive(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorAdaptive{s}, groundTruthRfs), 'omitnan');
    end

    [posBase, cardBase, ospaBase] = computeConsensusMetrics(stateEstimatesBySensorBase, modelBase);
    [posAdaptive, cardAdaptive, ospaAdaptive] = computeConsensusMetrics(stateEstimatesBySensorAdaptive, modelAdaptive);
    consOspaBase(trial) = mean(ospaBase);
    consOspaAdaptive(trial) = mean(ospaAdaptive);
    consPosBase(trial) = mean(posBase, 'omitnan');
    consPosAdaptive(trial) = mean(posAdaptive, 'omitnan');
    consCardBase(trial) = mean(cardBase);
    consCardAdaptive(trial) = mean(cardAdaptive);
end

fprintf('=====================================\n');
fprintf('GA Ideal Communication Comparison (N=%d)\n', numberOfTrials);
fprintf('Control=ordinary GA, Experiment=structure-aware decoupled KLA\n');
fprintf('=====================================\n');
fprintf('OSPA Consensus Error: %.6f -> %.6f\n', mean(consOspaBase), mean(consOspaAdaptive));
fprintf('Matched Localization Disagreement: %.6f -> %.6f\n', mean(consPosBase, 'omitnan'), mean(consPosAdaptive, 'omitnan'));
fprintf('Cardinality Dispersion: %.6f -> %.6f\n', mean(consCardBase), mean(consCardAdaptive));

summary.local.eOspaBase = mean(eOspaBase, 1);
summary.local.eOspaAdaptive = mean(eOspaAdaptive, 1);
summary.local.hOspaBase = mean(hOspaBase, 1);
summary.local.hOspaAdaptive = mean(hOspaAdaptive, 1);
summary.local.rmseBase = mean(rmseBase, 1);
summary.local.rmseAdaptive = mean(rmseAdaptive, 1);
summary.localTrials.eOspaBase = eOspaBase;
summary.localTrials.eOspaAdaptive = eOspaAdaptive;
summary.localTrials.hOspaBase = hOspaBase;
summary.localTrials.hOspaAdaptive = hOspaAdaptive;
summary.localTrials.rmseBase = rmseBase;
summary.localTrials.rmseAdaptive = rmseAdaptive;
summary.consensus.ospaBase = mean(consOspaBase);
summary.consensus.ospaAdaptive = mean(consOspaAdaptive);
summary.consensus.posBase = mean(consPosBase, 'omitnan');
summary.consensus.posAdaptive = mean(consPosAdaptive, 'omitnan');
summary.consensus.cardBase = mean(consCardBase);
summary.consensus.cardAdaptive = mean(consCardAdaptive);
summary.consensusTrials.ospaBase = consOspaBase;
summary.consensusTrials.ospaAdaptive = consOspaAdaptive;
summary.consensusTrials.posBase = consPosBase;
summary.consensusTrials.posAdaptive = consPosAdaptive;
summary.consensusTrials.cardBase = consCardBase;
summary.consensusTrials.cardAdaptive = consCardAdaptive;
summary.delta.ospa = summary.consensus.ospaAdaptive - summary.consensus.ospaBase;
summary.delta.pos = summary.consensus.posAdaptive - summary.consensus.posBase;
summary.delta.card = summary.consensus.cardAdaptive - summary.consensus.cardBase;
summary.delta.localEospa = mean(summary.local.eOspaAdaptive) - mean(summary.local.eOspaBase);
summary.delta.localHospa = mean(summary.local.hOspaAdaptive) - mean(summary.local.hOspaBase);
summary.delta.localRmse = mean(summary.local.rmseAdaptive) - mean(summary.local.rmseBase);
summary.pDropBySensorTrials = pDropBySensorTrials;
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);
summary.commConfig = commConfig;
summary.adaptiveFusionConfig = adaptiveFusionConfig;
summary.trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportName = sprintf('GA_IDEAL_COMM_COMPARE_N%d_SEED%d_%s.md', ...
        numberOfTrials, baseSeed, timestamp);
    reportPath = fullfile(reportDir, reportName);
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
        sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
        commConfig, pDropBySensorTrials, eOspaBase, eOspaAdaptive, hOspaBase, hOspaAdaptive, ...
        rmseBase, rmseAdaptive, consOspaBase, consOspaAdaptive, consPosBase, ...
        consPosAdaptive, consCardBase, consCardAdaptive);
    fprintf('Report written: %s\n', reportPath);
end
end

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
    sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
    commConfig, pDropBySensorTrials, eOspaBase, eOspaAdaptive, hOspaBase, hOspaAdaptive, ...
    rmseBase, rmseAdaptive, consOspaBase, consOspaAdaptive, consPosBase, consPosAdaptive, ...
    consCardBase, consCardAdaptive)

fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf(fid, '# GA Ideal Communication Comparison (%s)\n\n', timestamp);
fprintf(fid, 'Comparison order: ordinary GA -> structure-aware decoupled KLA\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
if useFixedSeed
    fprintf(fid, '- trialSeeds: %s\n', mat2str(computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)));
end
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- comm level: %d\n', getField(commConfig, 'level', 0));
fprintf(fid, '- linkModel: %s\n', getField(commConfig, 'linkModel', 'fixed'));
fprintf(fid, '- pDrop target mean: %.3f\n', getField(commConfig, 'pDrop', 0));
fprintf(fid, '- adaptive useDecoupledKla: %d\n', adaptiveFusionConfig.useDecoupledKla);
fprintf(fid, '- adaptive useStructureAwareKla: %d\n', adaptiveFusionConfig.useStructureAwareKla);
fprintf(fid, '- adaptive usePosteriorStructureConsistency: %d\n', getField(adaptiveFusionConfig, 'usePosteriorStructureConsistency', false));
fprintf(fid, '- adaptive spatialStructureStrength: %.3f\n', adaptiveFusionConfig.spatialStructureStrength);
fprintf(fid, '- adaptive existenceStructureStrength: %.3f\n', adaptiveFusionConfig.existenceStructureStrength);
fprintf(fid, '- adaptive structureReliabilityPower: %.3f\n\n', adaptiveFusionConfig.structureReliabilityPower);

fprintf(fid, '## Mean pDropBySensor Across Trials\n');
fprintf(fid, '- %s\n\n', mat2str(mean(pDropBySensorTrials, 1), 4));

fprintf(fid, '## Per-Trial Network Disagreement Metrics\n');
fprintf(fid, '| Trial | Seed | Arm | OSPA | RMSE | Cardinality |\n');
fprintf(fid, '|------:|-----:|:----|-----:|-----:|------------:|\n');
trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
for trial = 1:numberOfTrials
    fprintf(fid, '| %d | %.0f | ordinary GA | %.6f | %.6f | %.6f |\n', ...
        trial, trialSeeds(trial), consOspaBase(trial), consPosBase(trial), consCardBase(trial));
    fprintf(fid, '| %d | %.0f | structure-aware decoupled KLA | %.6f | %.6f | %.6f |\n', ...
        trial, trialSeeds(trial), consOspaAdaptive(trial), consPosAdaptive(trial), consCardAdaptive(trial));
end
fprintf(fid, '\n');

fprintf(fid, '## Per-Sensor Metrics (mean across trials)\n');
fprintf(fid, '| Sensor | E-OSPA (GA) | E-OSPA (SA) | H-OSPA (GA) | H-OSPA (SA) | RMSE (GA) | RMSE (SA) |\n');
fprintf(fid, '|:------:|------------:|------------:|------------:|------------:|----------:|----------:|\n');
numberOfSensors = size(eOspaBase, 2);
for s = 1:numberOfSensors
    fprintf(fid, '| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n', ...
        s, mean(eOspaBase(:, s)), mean(eOspaAdaptive(:, s)), ...
        mean(hOspaBase(:, s)), mean(hOspaAdaptive(:, s)), ...
        mean(rmseBase(:, s)), mean(rmseAdaptive(:, s)));
end
fprintf(fid, '\n');

fprintf(fid, '## Aggregated Local Metrics (mean across sensors and trials)\n');
fprintf(fid, '- E-OSPA: %.3f -> %.3f\n', mean(eOspaBase(:)), mean(eOspaAdaptive(:)));
fprintf(fid, '- H-OSPA: %.3f -> %.3f\n', mean(hOspaBase(:)), mean(hOspaAdaptive(:)));
fprintf(fid, '- RMSE: %.3f -> %.3f\n\n', mean(rmseBase(:)), mean(rmseAdaptive(:)));

fprintf(fid, '## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '- OSPA consensus error: %.3f -> %.3f\n', mean(consOspaBase), mean(consOspaAdaptive));
fprintf(fid, '- Matched localization disagreement: %.3f -> %.3f\n', mean(consPosBase, 'omitnan'), mean(consPosAdaptive, 'omitnan'));
fprintf(fid, '- Cardinality dispersion: %.3f -> %.3f\n', mean(consCardBase), mean(consCardAdaptive));

fprintf(fid, '\n## Network Disagreement Metrics With Trial Variability\n');
writeMetricStatsTable(fid, {'ordinary GA', 'structure-aware decoupled KLA'}, ...
    {'OSPA', 'RMSE', 'Cardinality'}, ...
    { [consOspaBase, consOspaAdaptive], [consPosBase, consPosAdaptive], [consCardBase, consCardAdaptive] }, ...
    [false, true, false]);

fprintf(fid, '\n## Paired Improvements Relative to Ordinary GA\n');
writePairedImprovementTable(fid, {'ordinary GA', 'structure-aware decoupled KLA'}, ...
    {'OSPA', 'RMSE', 'Cardinality'}, ...
    { [consOspaBase, consOspaAdaptive], [consPosBase, consPosAdaptive], [consCardBase, consCardAdaptive] }, ...
    [false, true, false]);

fprintf(fid, '\n## Local Tracking Metrics With Trial Variability\n');
localEOspaTrial = computeTrialSensorMeansPair(eOspaBase, eOspaAdaptive, false);
localHOspaTrial = computeTrialSensorMeansPair(hOspaBase, hOspaAdaptive, false);
localRmseTrial = computeTrialSensorMeansPair(rmseBase, rmseAdaptive, true);
writeMetricStatsTable(fid, {'ordinary GA', 'structure-aware decoupled KLA'}, ...
    {'E-OSPA', 'H-OSPA', 'RMSE'}, ...
    {localEOspaTrial.e, localHOspaTrial.e, localRmseTrial.e}, ...
    [false, false, true]);

fprintf(fid, '\n## Paired Local-Metric Improvements Relative to Ordinary GA\n');
writePairedImprovementTable(fid, {'ordinary GA', 'structure-aware decoupled KLA'}, ...
    {'E-OSPA', 'H-OSPA', 'RMSE'}, ...
    {localEOspaTrial.e, localHOspaTrial.e, localRmseTrial.e}, ...
    [false, false, true]);

fclose(fid);
end

function seeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)
if useFixedSeed
    seeds = baseSeed + (1:numberOfTrials);
else
    seeds = NaN(1, numberOfTrials);
end
end

function out = computeTrialSensorMeansPair(baseValues, adaptiveValues, omitnanFlag)
out.e = zeros(size(baseValues, 1), 2);
for trial = 1:size(baseValues, 1)
    out.e(trial, 1) = summarizeVector(reshape(baseValues(trial, :), [], 1), omitnanFlag).mean;
    out.e(trial, 2) = summarizeVector(reshape(adaptiveValues(trial, :), [], 1), omitnanFlag).mean;
end
end

function writeMetricStatsTable(fid, armNames, metricNames, metricArrays, omitnanFlags)
fprintf(fid, '| Arm | Metric | Mean +/- Std | 95%% CI | N |\n');
fprintf(fid, '|:----|:-------|-------------:|:-------|--:|\n');
for metricIdx = 1:numel(metricNames)
    data = metricArrays{metricIdx};
    omitnanFlag = omitnanFlags(metricIdx);
    for armIdx = 1:numel(armNames)
        stats = summarizeVector(data(:, armIdx), omitnanFlag);
        fprintf(fid, '| %s | %s | %.6f +/- %.6f | [%.6f, %.6f] | %d |\n', ...
            armNames{armIdx}, metricNames{metricIdx}, stats.mean, stats.std, ...
            stats.ciLow, stats.ciHigh, stats.n);
    end
end
end

function writePairedImprovementTable(fid, armNames, metricNames, metricArrays, omitnanFlags)
fprintf(fid, '| Arm | Metric | Paired reduction | 95%% CI | Reduction | Wins | Sign-test p |\n');
fprintf(fid, '|:----|:-------|-----------------:|:-------|----------:|-----:|------------:|\n');
baselineIdx = 1;
for metricIdx = 1:numel(metricNames)
    data = metricArrays{metricIdx};
    omitnanFlag = omitnanFlags(metricIdx);
    baseline = data(:, baselineIdx);
    candidate = data(:, 2);
    valid = isfinite(baseline) & isfinite(candidate);
    if ~omitnanFlag
        valid = valid & ~isnan(baseline) & ~isnan(candidate);
    end
    deltas = baseline(valid) - candidate(valid);
    stats = summarizeVector(deltas, true);
    baseStats = summarizeVector(baseline(valid), true);
    pct = 100 * stats.mean / baseStats.mean;
    wins = sum(deltas > 0);
    pValue = signTestPvalue(deltas);
    fprintf(fid, '| %s | %s | %.6f +/- %.6f | [%.6f, %.6f] | %.2f%% | %d/%d | %.4g |\n', ...
        armNames{2}, metricNames{metricIdx}, stats.mean, stats.std, ...
        stats.ciLow, stats.ciHigh, pct, wins, stats.n, pValue);
end
end

function stats = summarizeVector(values, omitnanFlag)
values = values(:);
if nargin >= 2 && omitnanFlag
    values = values(isfinite(values));
end
stats.n = numel(values);
if stats.n == 0
    stats.mean = NaN;
    stats.std = NaN;
    stats.ciLow = NaN;
    stats.ciHigh = NaN;
    return;
end
stats.mean = mean(values);
if stats.n > 1
    stats.std = std(values, 0);
    halfWidth = tCritical95(stats.n - 1) * stats.std / sqrt(stats.n);
else
    stats.std = 0;
    halfWidth = 0;
end
stats.ciLow = stats.mean - halfWidth;
stats.ciHigh = stats.mean + halfWidth;
end

function tcrit = tCritical95(df)
table = [ ...
    12.706, 4.303, 3.182, 2.776, 2.571, ...
    2.447, 2.365, 2.306, 2.262, 2.228, ...
    2.201, 2.179, 2.160, 2.145, 2.131, ...
    2.120, 2.110, 2.101, 2.093, 2.086, ...
    2.080, 2.074, 2.069, 2.064, 2.060, ...
    2.056, 2.052, 2.048, 2.045, 2.042];
if df < 1
    tcrit = 0;
elseif df <= numel(table)
    tcrit = table(df);
else
    tcrit = 1.96;
end
end

function pValue = signTestPvalue(deltas)
deltas = deltas(:);
deltas = deltas(isfinite(deltas) & abs(deltas) > eps);
n = numel(deltas);
if n == 0
    pValue = NaN;
    return;
end
wins = sum(deltas > 0);
k = min(wins, n - wins);
tail = 0;
for i = 0:k
    logProb = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1) - n * log(2);
    tail = tail + exp(logProb);
end
pValue = min(1, 2 * tail);
end

function merged = mergeStructFields(base, overrides)
merged = base;
if nargin < 2 || ~isstruct(overrides) || isempty(fieldnames(overrides))
    return;
end
fields = fieldnames(overrides);
for i = 1:numel(fields)
    merged.(fields{i}) = overrides.(fields{i});
end
end

function projectRoot = resolveProjectRoot(scriptDir)
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = scriptDir;
for k = 1:6
    if exist(fullfile(projectRoot, 'setPath.m'), 'file')
        return;
    end
    parent = fileparts(projectRoot);
    if isempty(parent) || strcmp(parent, projectRoot)
        break;
    end
    projectRoot = parent;
end
end

function sensorInitialStates = buildSensorInitialStates()
groupCenters = [-80, -80; 35, -35];
groupSpacing = 20;
formationType = 'Leader3';
formationSpeed = [0.8; 0];
sensorsPerGroup = 4;
numGroups = size(groupCenters, 2);
sensorInitialStates = cell(1, numGroups * sensorsPerGroup);
idx = 1;
for g = 1:numGroups
    offsets = localFormationOffsets(formationType, groupSpacing, sensorsPerGroup);
    center = groupCenters(:, g);
    for k = 1:sensorsPerGroup
        pos = center + offsets(:, k);
        sensorInitialStates{idx} = [pos; formationSpeed];
        idx = idx + 1;
    end
end
end

function neighborMap = buildNeighborMap4Plus4(numberOfSensors)
if numberOfSensors ~= 8
    error('buildNeighborMap4Plus4 expects numberOfSensors = 8.');
end
groupA = 1:4;
groupB = 5:8;
pairings = [1 5; 2 6; 3 7; 4 8];
neighborMap = cell(1, numberOfSensors);
for i = 1:4
    neighborMap{groupA(i)} = unique([groupA, pairings(i, 2)]);
    neighborMap{groupB(i)} = unique([groupB, pairings(i, 1)]);
end
end

function targetBirthStates = buildTargetBirthStates()
targetCenter = [0; 0];
groupCenters = [70, 80, 70; 80, 0, -80];
groupTypes = {'Triangle', 'Triangle', 'Leader3'};
groupCounts = [3, 3, 4];
groupSpacing = [30, 25, 20];
groupSpeed = [0.45, 0.45, 0.45];
totalTargets = sum(groupCounts);
targetBirthStates = zeros(4, totalTargets);
idx = 1;
for g = 1:numel(groupCounts)
    offsets = localFormationOffsets(groupTypes{g}, groupSpacing(g), groupCounts(g));
    center = groupCenters(:, g);
    dir = targetCenter - center;
    if norm(dir) < 1e-6
        dir = [-1; 0];
    end
    vel = (groupSpeed(g) / norm(dir)) * dir;
    for k = 1:groupCounts(g)
        pos = center + offsets(:, k);
        targetBirthStates(:, idx) = [pos; vel];
        idx = idx + 1;
    end
end
end

function [posConsensus, cardConsensus, ospaConsensus] = computeConsensusMetrics(stateEstimatesBySensor, model)
numSensors = numel(stateEstimatesBySensor);
simLength = numel(stateEstimatesBySensor{1}.mu);
posConsensus = zeros(1, simLength);
cardConsensus = zeros(1, simLength);
ospaConsensus = zeros(1, simLength);
for t = 1:simLength
    counts = zeros(1, numSensors);
    for s = 1:numSensors
        counts(s) = numel(stateEstimatesBySensor{s}.mu{t});
    end
    medCount = median(counts);
    cardConsensus(t) = mean(abs(counts - medCount));
    pairSum = 0;
    pairCount = 0;
    ospaSum = 0;
    ospaCount = 0;
    for i = 1:numSensors-1
        for j = i+1:numSensors
            d = estimateSetRmse(stateEstimatesBySensor{i}, stateEstimatesBySensor{j}, t);
            if ~isnan(d)
                pairSum = pairSum + d;
                pairCount = pairCount + 1;
            end
            dOspa = estimateSetOspaConsensus(stateEstimatesBySensor{i}, stateEstimatesBySensor{j}, t, model);
            ospaSum = ospaSum + dOspa;
            ospaCount = ospaCount + 1;
        end
    end
    posConsensus(t) = ternary(pairCount > 0, pairSum / pairCount, NaN);
    ospaConsensus(t) = ternary(ospaCount > 0, ospaSum / ospaCount, NaN);
end
end

function dist = estimateSetRmse(estA, estB, t)
muA = estA.mu{t};
muB = estB.mu{t};
if isempty(muA) && isempty(muB)
    dist = 0;
    return;
end
if isempty(muA) || isempty(muB)
    dist = NaN;
    return;
end
XA = cell2mat(cellfun(@(x) x(1:2), muA, 'UniformOutput', false));
XB = cell2mat(cellfun(@(x) x(1:2), muB, 'UniformOutput', false));
n = size(XA, 2);
m = size(XB, 2);
if n == 0 || m == 0
    dist = NaN;
    return;
end
D = zeros(n, m);
for i = 1:n
    for j = 1:m
        d = XA(:, i) - XB(:, j);
        D(i, j) = sqrt(d' * d);
    end
end
[matching, ~] = Hungarian(D);
matched = D(matching == 1);
dist = ternary(isempty(matched), NaN, sqrt(mean(matched.^2)));
end

function rmseSeries = computeSetRmseOverTime(stateEstimates, groundTruthRfs)
simLength = numel(groundTruthRfs.x);
rmseSeries = NaN(1, simLength);
for t = 1:simLength
    rmseSeries(t) = computeSetRmseAtTime(stateEstimates, groundTruthRfs, t);
end
end

function rmse = computeSetRmseAtTime(stateEstimates, groundTruthRfs, t)
truthCells = groundTruthRfs.x{t};
estCells = stateEstimates.mu{t};
if isempty(truthCells) && isempty(estCells)
    rmse = 0;
    return;
end
if isempty(truthCells) || isempty(estCells)
    rmse = NaN;
    return;
end
XT = cell2mat(cellfun(@(x) x(1:2), truthCells, 'UniformOutput', false));
XE = cell2mat(cellfun(@(x) x(1:2), estCells, 'UniformOutput', false));
n = size(XT, 2);
m = size(XE, 2);
if n == 0 || m == 0
    rmse = NaN;
    return;
end
D = zeros(n, m);
for i = 1:n
    for j = 1:m
        d = XT(:, i) - XE(:, j);
        D(i, j) = sqrt(d' * d);
    end
end
[matching, ~] = Hungarian(D);
matched = D(matching == 1);
rmse = ternary(isempty(matched), NaN, sqrt(mean(matched.^2)));
end

function dist = estimateSetOspaConsensus(estA, estB, t, model)
muA = estA.mu{t};
SigmaA = estA.Sigma{t};
muB = estB.mu{t};
SigmaB = estB.Sigma{t};
if isempty(muA) && isempty(muB)
    dist = 0;
    return;
end
if isempty(muA) || isempty(muB)
    dist = model.ospaParameters.eC;
    return;
end
[eAB, ~] = ospa(muA, muA, SigmaA, muB, SigmaB, model.ospaParameters);
[eBA, ~] = ospa(muB, muB, SigmaB, muA, SigmaA, model.ospaParameters);
dist = 0.5 * (eAB(1) + eBA(1));
end

function offsets = localFormationOffsets(formationType, spacing, count)
if nargin < 3
    count = 3;
end

switch lower(formationType)
    case 'triangle'
        baseOffsets = [0, -0.5, 0.5; 0.577, -0.289, -0.289];
    case 'leader3'
        baseOffsets = [0, -0.75, -0.75, -1.5; 0, 0.6, -0.6, 0];
    otherwise
        baseOffsets = zeros(2, count);
end

if size(baseOffsets, 2) < count
    repeats = ceil(count / size(baseOffsets, 2));
    baseOffsets = repmat(baseOffsets, 1, repeats);
end
offsets = spacing * baseOffsets(:, 1:count);
end

function value = getField(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function result = ternary(condition, trueValue, falseValue)
if condition
    result = trueValue;
else
    result = falseValue;
end
end
