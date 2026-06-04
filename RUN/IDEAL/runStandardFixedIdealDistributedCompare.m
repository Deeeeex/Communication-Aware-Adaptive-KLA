function [reportPath, summary] = runStandardFixedIdealDistributedCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport)
% RUNSTANDARDFIXEDIDEALDISTRIBUTEDCOMPARE
% Compare fixed distributed GA against the current-best adaptive GA on the
% standard fixed scenario under ideal communication and a 4-sensor ring graph.

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

reportPath = '';
summary = struct();

numberOfSensors = 4;
clutterRates = 5 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);
scenarioType = 'Fixed';
fusionWeighting = 'Uniform';
topologyName = 'ring';
neighborMap = buildRingNeighborMap(numberOfSensors);

commConfig = struct();
commConfig.level = 0;
commConfig.globalMaxMeasurementsPerStep = inf;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.0;
commConfig.pDropBySensor = zeros(1, numberOfSensors);

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
    'spatialStructureStrength', 0.35, ...
    'existenceStructureStrength', 0.05, ...
    'structureReliabilityPower', 0.25, ...
    'structureReliabilityMinScore', 0.25, ...
    'useFreshness', false, ...
    'useHistory', false, ...
    'useNIS', false, ...
    'robustNIS', false);

eOspaBaseline = zeros(numberOfTrials, numberOfSensors);
rmseBaseline = zeros(numberOfTrials, numberOfSensors);
eOspaAdaptive = zeros(numberOfTrials, numberOfSensors);
rmseAdaptive = zeros(numberOfTrials, numberOfSensors);
consOspaBaseline = zeros(numberOfTrials, 1);
consPosBaseline = zeros(numberOfTrials, 1);
consCardBaseline = zeros(numberOfTrials, 1);
consOspaAdaptive = zeros(numberOfTrials, 1);
consPosAdaptive = zeros(numberOfTrials, 1);
consCardAdaptive = zeros(numberOfTrials, 1);
pDropBySensorTrials = zeros(numberOfTrials, numberOfSensors);
structureSpatialPriors = cell(1, numberOfSensors);
structureExistencePriors = cell(1, numberOfSensors);

for trial = 1:numberOfTrials
    fprintf('Trial %d/%d\n', trial, numberOfTrials);
    if useFixedSeed
        rng(baseSeed + trial);
    end

    model = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'GA', 'LBP', scenarioType);
    model.simulationLength = 100;
    model.fusionWeighting = fusionWeighting;

    [~, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
    [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
    pDropBySensorTrials(trial, :) = reshape(commStats.pDropBySensor, 1, []);

    modelBaseline = model;
    modelBaseline.adaptiveFusion = adaptiveFusionConfig;
    modelBaseline.adaptiveFusion.enabled = false;
    [stateBaseline, localModelsBaseline] = runDistributedLmbFilter( ...
        modelBaseline, measurementsDelivered, [], neighborMap, commStats);

    modelAdaptive = model;
    modelAdaptive.adaptiveFusion = adaptiveFusionConfig;
    modelAdaptive.adaptiveFusion.enabled = true;
    [stateAdaptive, localModelsAdaptive] = runDistributedLmbFilter( ...
        modelAdaptive, measurementsDelivered, [], neighborMap, commStats);

    if trial == 1
        for s = 1:numberOfSensors
            structureSpatialPriors{s} = localModelsAdaptive{s}.gaSpatialStructurePrior;
            structureExistencePriors{s} = localModelsAdaptive{s}.gaExistenceStructurePrior;
        end
    end

    for s = 1:numberOfSensors
        [eBase, ~] = computeSimulationOspa(localModelsBaseline{s}, groundTruthRfs, stateBaseline{s});
        [eAda, ~] = computeSimulationOspa(localModelsAdaptive{s}, groundTruthRfs, stateAdaptive{s});
        eOspaBaseline(trial, s) = mean(eBase);
        eOspaAdaptive(trial, s) = mean(eAda);
        rmseBaseline(trial, s) = mean(computeSetRmseOverTime(stateBaseline{s}, groundTruthRfs), 'omitnan');
        rmseAdaptive(trial, s) = mean(computeSetRmseOverTime(stateAdaptive{s}, groundTruthRfs), 'omitnan');
    end

    [posBase, cardBase, ospaBase] = computeConsensusMetrics(stateBaseline, modelBaseline);
    [posAda, cardAda, ospaAda] = computeConsensusMetrics(stateAdaptive, modelAdaptive);
    consOspaBaseline(trial) = mean(ospaBase);
    consPosBaseline(trial) = mean(posBase, 'omitnan');
    consCardBaseline(trial) = mean(cardBase);
    consOspaAdaptive(trial) = mean(ospaAda);
    consPosAdaptive(trial) = mean(posAda, 'omitnan');
    consCardAdaptive(trial) = mean(cardAda);
end

summary.config = struct( ...
    'numberOfSensors', numberOfSensors, ...
    'scenarioType', scenarioType, ...
    'simulationLength', 100, ...
    'fusionWeighting', fusionWeighting, ...
    'topologyName', topologyName, ...
    'neighborMap', {neighborMap}, ...
    'clutterRates', clutterRates, ...
    'detectionProbabilities', detectionProbabilities, ...
    'measurementNoiseStd', q);
summary.baseline.eOspa = mean(eOspaBaseline, 1);
summary.baseline.rmse = mean(rmseBaseline, 1);
summary.adaptive.eOspa = mean(eOspaAdaptive, 1);
summary.adaptive.rmse = mean(rmseAdaptive, 1);
summary.consensus.ospaBaseline = mean(consOspaBaseline);
summary.consensus.posBaseline = mean(consPosBaseline, 'omitnan');
summary.consensus.cardBaseline = mean(consCardBaseline);
summary.consensus.ospaAdaptive = mean(consOspaAdaptive);
summary.consensus.posAdaptive = mean(consPosAdaptive, 'omitnan');
summary.consensus.cardAdaptive = mean(consCardAdaptive);
summary.delta = struct( ...
    'ospa', summary.consensus.ospaAdaptive - summary.consensus.ospaBaseline, ...
    'pos', summary.consensus.posAdaptive - summary.consensus.posBaseline, ...
    'card', summary.consensus.cardAdaptive - summary.consensus.cardBaseline, ...
    'localEospa', mean(summary.adaptive.eOspa) - mean(summary.baseline.eOspa), ...
    'localRmse', mean(summary.adaptive.rmse) - mean(summary.baseline.rmse));
summary.structure = struct( ...
    'spatialPriors', {structureSpatialPriors}, ...
    'existencePriors', {structureExistencePriors});
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);
summary.pDropBySensorTrials = pDropBySensorTrials;
summary.commConfig = commConfig;

fprintf('=====================================\n');
fprintf('Standard Ideal Distributed Compare (N=%d)\n', numberOfTrials);
fprintf('Baseline=fixed distributed GA, Experiment=current-best adaptive distributed GA\n');
fprintf('=====================================\n');
fprintf('OSPA consensus error: %.6f -> %.6f\n', summary.consensus.ospaBaseline, summary.consensus.ospaAdaptive);
fprintf('Matched localization disagreement: %.6f -> %.6f\n', summary.consensus.posBaseline, summary.consensus.posAdaptive);
fprintf('Cardinality dispersion: %.6f -> %.6f\n', summary.consensus.cardBaseline, summary.consensus.cardAdaptive);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'IDEAL');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPath = fullfile(reportDir, sprintf('STANDARD_FIXED_IDEAL_DISTRIBUTED_%s.md', timestamp));
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, summary);
    fprintf('Report written: %s\n', reportPath);
end
end

function neighborMap = buildRingNeighborMap(numberOfSensors)
neighborMap = cell(1, numberOfSensors);
for s = 1:numberOfSensors
    left = mod(s - 2, numberOfSensors) + 1;
    right = mod(s, numberOfSensors) + 1;
    neighborMap{s} = [left, s, right];
end
end

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, summary)
fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Standard Fixed Ideal Distributed Compare (%s)\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Ideal communication, 4-sensor ring graph, standard `Fixed` scenario.\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
fprintf(fid, '- topology: %s\n', summary.config.topologyName);
fprintf(fid, '- neighborMap: %s\n', mat2str(cell2mat(cellfun(@(x) reshape(x, 1, []), summary.config.neighborMap, 'UniformOutput', false)')));
fprintf(fid, '- ideal pDropBySensor: %s\n\n', mat2str(summary.meanPDropBySensor, 4));

fprintf(fid, '## Network Disagreement Metrics\n');
fprintf(fid, '| Mode | OSPA | RMSE | Card |\n');
fprintf(fid, '|:-----|-----:|-----:|-----:|\n');
fprintf(fid, '| Fixed distributed GA | %.6f | %.6f | %.6f |\n', ...
    summary.consensus.ospaBaseline, summary.consensus.posBaseline, summary.consensus.cardBaseline);
fprintf(fid, '| Current-best adaptive distributed GA | %.6f | %.6f | %.6f |\n\n', ...
    summary.consensus.ospaAdaptive, summary.consensus.posAdaptive, summary.consensus.cardAdaptive);

fprintf(fid, '## Structure Priors\n');
for s = 1:numel(summary.structure.spatialPriors)
    fprintf(fid, '- Sensor %d spatial prior: %s\n', s, mat2str(summary.structure.spatialPriors{s}, 4));
    fprintf(fid, '- Sensor %d existence prior: %s\n', s, mat2str(summary.structure.existencePriors{s}, 4));
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

function result = ternary(condition, trueValue, falseValue)
if condition
    result = trueValue;
else
    result = falseValue;
end
end

function projectRoot = resolveProjectRoot(scriptDir)
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
