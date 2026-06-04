function [reportPath, summary] = runMultisensorFilters_standard_fixed_ideal_compare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport)
% RUNMULTISENSORFILTERS_STANDARD_FIXED_IDEAL_COMPARE
% Simple ideal-communication fixed-weight GA/AA comparison aligned with the
% standard fixed multi-object scenario used in common LMB benchmarks.

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

commConfig = struct();
commConfig.level = 0;
commConfig.globalMaxMeasurementsPerStep = inf;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.0;
commConfig.pDropBySensor = zeros(1, numberOfSensors);

eOspaGa = zeros(numberOfTrials, 1);
hOspaGa = zeros(numberOfTrials, 1);
rmseGa = zeros(numberOfTrials, 1);
cardErrorGa = zeros(numberOfTrials, 1);

eOspaAa = zeros(numberOfTrials, 1);
hOspaAa = zeros(numberOfTrials, 1);
rmseAa = zeros(numberOfTrials, 1);
cardErrorAa = zeros(numberOfTrials, 1);

pDropBySensorTrials = zeros(numberOfTrials, numberOfSensors);

for trial = 1:numberOfTrials
    fprintf('Trial %d/%d\n', trial, numberOfTrials);
    if useFixedSeed
        rng(baseSeed + trial);
    end

    baseModel = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'GA', 'LBP', scenarioType);
    baseModel.simulationLength = 100;
    baseModel.gaSensorWeights = ones(1, numberOfSensors) / numberOfSensors;
    baseModel.aaSensorWeights = ones(1, numberOfSensors) / numberOfSensors;
    baseModel.adaptiveFusion = struct('enabled', false);
    baseModel.fusionWeighting = fusionWeighting;

    [~, measurements, groundTruthRfs] = generateMultisensorGroundTruth(baseModel);
    [measurementsDelivered, commStats] = applyCommunicationModel(measurements, baseModel, commConfig);
    pDropBySensorTrials(trial, :) = reshape(commStats.pDropBySensor, 1, []);

    modelGa = baseModel;
    modelGa.lmbParallelUpdateMode = 'GA';
    stateEstimatesGa = runParallelUpdateLmbFilter(modelGa, measurementsDelivered, commStats, []);
    [eGa, hGa, cardGa] = computeSimulationOspa(modelGa, groundTruthRfs, stateEstimatesGa);
    eOspaGa(trial) = mean(eGa);
    hOspaGa(trial) = mean(hGa);
    rmseGa(trial) = mean(computeSetRmseOverTime(stateEstimatesGa, groundTruthRfs), 'omitnan');
    cardErrorGa(trial) = mean(abs(cardGa - groundTruthRfs.cardinality));

    modelAa = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'AA', 'LBP', scenarioType);
    modelAa.simulationLength = baseModel.simulationLength;
    modelAa.gaSensorWeights = baseModel.gaSensorWeights;
    modelAa.aaSensorWeights = baseModel.aaSensorWeights;
    modelAa.adaptiveFusion = baseModel.adaptiveFusion;
    modelAa.fusionWeighting = fusionWeighting;
    stateEstimatesAa = runParallelUpdateLmbFilter(modelAa, measurementsDelivered, commStats, []);
    [eAa, hAa, cardAa] = computeSimulationOspa(modelAa, groundTruthRfs, stateEstimatesAa);
    eOspaAa(trial) = mean(eAa);
    hOspaAa(trial) = mean(hAa);
    rmseAa(trial) = mean(computeSetRmseOverTime(stateEstimatesAa, groundTruthRfs), 'omitnan');
    cardErrorAa(trial) = mean(abs(cardAa - groundTruthRfs.cardinality));
end

summary.config = struct( ...
    'numberOfSensors', numberOfSensors, ...
    'scenarioType', scenarioType, ...
    'simulationLength', 100, ...
    'fusionWeighting', fusionWeighting, ...
    'clutterRates', clutterRates, ...
    'detectionProbabilities', detectionProbabilities, ...
    'measurementNoiseStd', q);
summary.ga = buildMetricSummary(eOspaGa, hOspaGa, rmseGa, cardErrorGa);
summary.aa = buildMetricSummary(eOspaAa, hOspaAa, rmseAa, cardErrorAa);
summary.delta.gaMinusAa = struct( ...
    'eOspa', summary.ga.eOspa - summary.aa.eOspa, ...
    'hOspa', summary.ga.hOspa - summary.aa.hOspa, ...
    'rmse', summary.ga.rmse - summary.aa.rmse, ...
    'cardError', summary.ga.cardError - summary.aa.cardError);
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);
summary.pDropBySensorTrials = pDropBySensorTrials;
summary.commConfig = commConfig;

fprintf('=====================================\n');
fprintf('Standard Ideal Fixed-Weight GA/AA Comparison (N=%d)\n', numberOfTrials);
fprintf('Reference alignment: 4 sensors, ideal communication, uniform weights, Fixed scenario\n');
fprintf('=====================================\n');
fprintf('GA: E-OSPA %.6f, H-OSPA %.6f, RMSE %.6f, CardErr %.6f\n', ...
    summary.ga.eOspa, summary.ga.hOspa, summary.ga.rmse, summary.ga.cardError);
fprintf('AA: E-OSPA %.6f, H-OSPA %.6f, RMSE %.6f, CardErr %.6f\n', ...
    summary.aa.eOspa, summary.aa.hOspa, summary.aa.rmse, summary.aa.cardError);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'IDEAL');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPath = fullfile(reportDir, sprintf('STANDARD_FIXED_IDEAL_GA_AA_%s.md', timestamp));
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, summary);
    fprintf('Report written: %s\n', reportPath);
end
end

function metrics = buildMetricSummary(eOspa, hOspa, rmse, cardError)
metrics = struct();
metrics.eOspa = mean(eOspa);
metrics.hOspa = mean(hOspa);
metrics.rmse = mean(rmse, 'omitnan');
metrics.cardError = mean(cardError);
metrics.trials = struct('eOspa', eOspa, 'hOspa', hOspa, 'rmse', rmse, 'cardError', cardError);
end

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, summary)
fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end

cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '# Standard Fixed Ideal GA/AA Comparison (%s)\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'This run approximates the ideal setup emphasized in Li et al. (2023): ');
fprintf(fid, '4 synchronous sensors, ideal communication, uniform fusion weights, and full access to the common ROI. ');
fprintf(fid, 'The target ground truth follows the repo''s standard `Fixed` benchmark scenario inherited from the common LMB codebase.\n\n');

fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
fprintf(fid, '- numberOfSensors: %d\n', summary.config.numberOfSensors);
fprintf(fid, '- scenarioType: %s\n', summary.config.scenarioType);
fprintf(fid, '- simulationLength: %d\n', summary.config.simulationLength);
fprintf(fid, '- fusionWeighting: %s\n', summary.config.fusionWeighting);
fprintf(fid, '- clutterRates: %s\n', mat2str(summary.config.clutterRates));
fprintf(fid, '- detectionProbabilities: %s\n', mat2str(summary.config.detectionProbabilities));
fprintf(fid, '- measurementNoiseStd: %s\n', mat2str(summary.config.measurementNoiseStd));
fprintf(fid, '- ideal pDropBySensor: %s\n\n', mat2str(summary.meanPDropBySensor, 4));

fprintf(fid, '## Mean Metrics\n');
fprintf(fid, '| Filter | E-OSPA | H-OSPA | RMSE | Cardinality Error |\n');
fprintf(fid, '|:------|-------:|-------:|-----:|------------------:|\n');
fprintf(fid, '| GA | %.6f | %.6f | %.6f | %.6f |\n', ...
    summary.ga.eOspa, summary.ga.hOspa, summary.ga.rmse, summary.ga.cardError);
fprintf(fid, '| AA | %.6f | %.6f | %.6f | %.6f |\n\n', ...
    summary.aa.eOspa, summary.aa.hOspa, summary.aa.rmse, summary.aa.cardError);

fprintf(fid, '## Delta (GA - AA)\n');
fprintf(fid, '- E-OSPA: %.6f\n', summary.delta.gaMinusAa.eOspa);
fprintf(fid, '- H-OSPA: %.6f\n', summary.delta.gaMinusAa.hOspa);
fprintf(fid, '- RMSE: %.6f\n', summary.delta.gaMinusAa.rmse);
fprintf(fid, '- Cardinality Error: %.6f\n', summary.delta.gaMinusAa.cardError);
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
if isempty(matched)
    rmse = NaN;
else
    rmse = sqrt(mean(matched.^2));
end
end
