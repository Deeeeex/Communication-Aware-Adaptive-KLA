function [reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkFreshnessCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, adaptiveFusionOverrides, commConfigOverrides, writeReport)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_TIEREDLINKFRESHNESSCOMPARE
% Compare the robust-NIS baseline against the same setup with an
% additional freshness factor under tiered per-sensor packet-drop rates.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

%% Trial config
if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 20;
end
if nargin < 2 || isempty(baseSeed)
    baseSeed = 1;
end
if nargin < 3 || isempty(useFixedSeed)
    useFixedSeed = true;
end
if nargin < 4 || isempty(adaptiveFusionOverrides)
    adaptiveFusionOverrides = struct();
end
if nargin < 5 || isempty(commConfigOverrides)
    commConfigOverrides = struct();
end
if nargin < 6 || isempty(writeReport)
    writeReport = true;
end

reportPath = '';
summary = struct();

%% Scenario switches
staggeredBirths = true;
leaderSensor = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis';
adaptiveFusionBaseConfig = struct( ...
    'enabled', true, ...
    'emaAlpha', 0.7, ...
    'minWeight', 0.05, ...
    'useHistory', false, ...
    'historyEmaAlpha', 0.8, ...
    'historyScale', 2.0, ...
    'historyMinScore', 0.4, ...
    'historyCovWeight', 0.4, ...
    'historyInnovationWeight', 0.4, ...
    'historyCardinalityWeight', 0.2, ...
    'useFreshness', false, ...
    'freshnessDecay', 0.5, ...
    'freshnessMinScore', 0.4, ...
    'nisQuantileEnabled', true, ...
    'nisQuantile', 0.7, ...
    'nisConsistencyConfidence', 0.7, ...
    'nisPenaltyScale', 4.0, ...
    'nisPenaltyMin', 0.3, ...
    'nisPenaltyLowerScale', 1.0, ...
    'nisPenaltyUpperScale', 6.0, ...
    'nisPenaltyLowerPower', 2.0, ...
    'nisPenaltyUpperPower', 2.0, ...
    'nisEmaEnabled', true, ...
    'nisEmaAlpha', 0.7);
robustNISMin = 0.3;

%% Sensor configuration
numberOfSensors = 8;
clutterRates = 3 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);

commConfig = struct();
commConfig.level = 2;
commConfig.globalMaxMeasurementsPerStep = 80;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
commConfig.pDropLevelCounts = [1, 4, 1, 2];
commConfig.maxOutageNodes = 1;

adaptiveFusionBaseConfig = mergeStructFields(adaptiveFusionBaseConfig, adaptiveFusionOverrides);
commConfig = mergeStructFields(commConfig, commConfigOverrides);

controlAdaptiveConfig = adaptiveFusionBaseConfig;
controlAdaptiveConfig.useNIS = true;
controlAdaptiveConfig.robustNIS = true;
controlAdaptiveConfig.robustNISMin = robustNISMin;
controlAdaptiveConfig.useFreshness = false;

experimentAdaptiveConfig = controlAdaptiveConfig;
experimentAdaptiveConfig.useFreshness = true;

%% Sensor formations
sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.0;
sensorMotionConfig.initialStates = buildSensorInitialStates();

%% Target formations
targetFormationConfig = struct();
targetFormationConfig.targetFormationEnabled = true;
targetFormationConfig.targetFormationStaggeredBirths = staggeredBirths;
targetFormationConfig.targetFormationBirthInterval = 8;
targetFormationConfig.targetFormationStartTime = 1;
targetFormationConfig.targetFormationLifeSpan = 100;
targetFormationConfig.targetBirthStates = buildTargetBirthStates();
targetFormationConfig.targetFormationCount = size(targetFormationConfig.targetBirthStates, 2);

%% Allocate trial results
eOspaControl = zeros(numberOfTrials, numberOfSensors);
eOspaExperiment = zeros(numberOfTrials, numberOfSensors);
rmseControl = zeros(numberOfTrials, numberOfSensors);
rmseExperiment = zeros(numberOfTrials, numberOfSensors);
consOspaControl = zeros(numberOfTrials, 1);
consOspaExperiment = zeros(numberOfTrials, 1);
consPosControl = zeros(numberOfTrials, 1);
consPosExperiment = zeros(numberOfTrials, 1);
consCardControl = zeros(numberOfTrials, 1);
consCardExperiment = zeros(numberOfTrials, 1);
pDropBySensorTrials = zeros(numberOfTrials, numberOfSensors);

%% Trials
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

    modelControl = model;
    modelControl.adaptiveFusion = controlAdaptiveConfig;
    controlNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorControl, localModelsControl] = runDistributedLmbFilter( ...
        modelControl, measurementsDelivered, sensorTrajectories, controlNeighborMap, commStats);

    modelExperiment = model;
    modelExperiment.adaptiveFusion = experimentAdaptiveConfig;
    experimentNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorExperiment, localModelsExperiment] = runDistributedLmbFilter( ...
        modelExperiment, measurementsDelivered, sensorTrajectories, experimentNeighborMap, commStats);

    for s = 1:numberOfSensors
        [eControl, ~] = computeSimulationOspa(localModelsControl{s}, groundTruthRfs, stateEstimatesBySensorControl{s});
        [eExperiment, ~] = computeSimulationOspa(localModelsExperiment{s}, groundTruthRfs, stateEstimatesBySensorExperiment{s});
        eOspaControl(trial, s) = mean(eControl);
        eOspaExperiment(trial, s) = mean(eExperiment);
        rmseControl(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorControl{s}, groundTruthRfs), 'omitnan');
        rmseExperiment(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorExperiment{s}, groundTruthRfs), 'omitnan');
    end

    [posControl, cardControl, ospaControl] = computeConsensusMetrics(stateEstimatesBySensorControl, modelControl);
    [posExperiment, cardExperiment, ospaExperiment] = computeConsensusMetrics(stateEstimatesBySensorExperiment, modelExperiment);
    consOspaControl(trial) = mean(ospaControl);
    consOspaExperiment(trial) = mean(ospaExperiment);
    consPosControl(trial) = mean(posControl, 'omitnan');
    consPosExperiment(trial) = mean(posExperiment, 'omitnan');
    consCardControl(trial) = mean(cardControl);
    consCardExperiment(trial) = mean(cardExperiment);
end

%% Summary
fprintf('=====================================\n');
fprintf('GA Tiered Link Freshness Comparison (N=%d)\n', numberOfTrials);
fprintf('Control=robust NIS baseline, Experiment=control + freshness\n');
fprintf('Tier levels=%s, counts=%s\n', mat2str(commConfig.pDropLevels, 3), mat2str(commConfig.pDropLevelCounts));
fprintf('=====================================\n');
for s = 1:numberOfSensors
    fprintf('Sensor %d: E-OSPA %.3f -> %.3f, RMSE %.3f -> %.3f, mean pDrop %.3f\n', ...
        s, mean(eOspaControl(:, s)), mean(eOspaExperiment(:, s)), ...
        mean(rmseControl(:, s)), mean(rmseExperiment(:, s)), mean(pDropBySensorTrials(:, s)));
end
fprintf('=====================================\n');
fprintf('Network Disagreement Metrics (robust NIS baseline -> + freshness)\n');
fprintf('=====================================\n');
fprintf('OSPA consensus error: %.3f -> %.3f\n', ...
    mean(consOspaControl), mean(consOspaExperiment));
fprintf('Matched localization disagreement: %.3f -> %.3f\n', ...
    mean(consPosControl, 'omitnan'), mean(consPosExperiment, 'omitnan'));
fprintf('Cardinality dispersion: %.3f -> %.3f\n', ...
    mean(consCardControl), mean(consCardExperiment));

summary.consensus.ospaControl = mean(consOspaControl);
summary.consensus.ospaExperiment = mean(consOspaExperiment);
summary.consensus.posControl = mean(consPosControl, 'omitnan');
summary.consensus.posExperiment = mean(consPosExperiment, 'omitnan');
summary.consensus.cardControl = mean(consCardControl);
summary.consensus.cardExperiment = mean(consCardExperiment);
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPath = fullfile(reportDir, sprintf('Del_GA_TIERED_LINK_FRESHNESS_COMPARE_%s.md', timestamp));
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, sensorCommRange, ...
        fusionWeighting, leaderSensor, controlAdaptiveConfig, experimentAdaptiveConfig, commConfig, ...
        pDropBySensorTrials, consOspaControl, consOspaExperiment, consPosControl, consPosExperiment, ...
        consCardControl, consCardExperiment);
    fprintf('Report written: %s\n', reportPath);
end
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

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, sensorCommRange, ...
    fusionWeighting, leaderSensor, controlAdaptiveConfig, experimentAdaptiveConfig, commConfig, ...
    pDropBySensorTrials, consOspaControl, consOspaExperiment, consPosControl, consPosExperiment, ...
    consCardControl, consCardExperiment)
fid = fopen(reportPath, 'w');
if fid < 0
    return;
end
fprintf(fid, '# GA Tiered Link Freshness Comparison (%s)\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Comparison order: robust NIS baseline -> robust NIS + freshness\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- pDropLevels: %s\n', mat2str(getField(commConfig, 'pDropLevels', []), 3));
fprintf(fid, '- pDropLevelCounts: %s\n', mat2str(getField(commConfig, 'pDropLevelCounts', [])));
fprintf(fid, '- control useNIS: %d\n', controlAdaptiveConfig.useNIS);
fprintf(fid, '- control robustNIS: %d\n', controlAdaptiveConfig.robustNIS);
fprintf(fid, '- experiment useFreshness: %d\n\n', experimentAdaptiveConfig.useFreshness);
fprintf(fid, '## Per-Trial pDropBySensor\n');
for trial = 1:size(pDropBySensorTrials, 1)
    fprintf(fid, '- Trial %d: %s\n', trial, mat2str(pDropBySensorTrials(trial, :), 4));
end
fprintf(fid, '\n## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '- OSPA consensus error: %.3f -> %.3f\n', mean(consOspaControl), mean(consOspaExperiment));
fprintf(fid, '- Matched localization disagreement: %.3f -> %.3f\n', mean(consPosControl, 'omitnan'), mean(consPosExperiment, 'omitnan'));
fprintf(fid, '- Cardinality dispersion: %.3f -> %.3f\n', mean(consCardControl), mean(consCardExperiment));
fclose(fid);
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
    posConsensus(t) = ternary(pairCount > 0, pairSum / max(pairCount, 1), NaN);
    ospaConsensus(t) = ternary(ospaCount > 0, ospaSum / max(ospaCount, 1), NaN);
end
end

function value = ternary(cond, a, b)
if cond
    value = a;
else
    value = b;
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
D = zeros(size(XA, 2), size(XB, 2));
for i = 1:size(XA, 2)
    for j = 1:size(XB, 2)
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
D = zeros(size(XT, 2), size(XE, 2));
for i = 1:size(XT, 2)
    for j = 1:size(XE, 2)
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
if nargin < 2
    spacing = 12;
end
if nargin < 1 || isempty(formationType)
    formationType = 'Triangle';
end
formationType = upper(formationType);
switch formationType
    case 'TRIANGLE'
        radius = spacing / sqrt(3);
        angles = (0:2) * 2 * pi / 3;
        baseOffsets = radius * [cos(angles); sin(angles)];
    case 'LINE'
        idx = (0:count-1) - (count-1)/2;
        baseOffsets = [zeros(1, count); idx * spacing];
    case 'LEADER3'
        y1 = 0.7 * spacing;
        baseOffsets = [0, -spacing, -spacing, -2*spacing;
                       0,  y1,       -y1,       0];
    case 'LEADER4'
        y1 = 0.7 * spacing;
        y2 = 1.4 * spacing;
        baseOffsets = [0, -spacing, -spacing, -2*spacing, -2*spacing;
                       0,  y1,       -y1,       y2,        -y2];
    case 'CIRCLE'
        angles = 2 * pi * (0:count-1) / max(count, 1);
        baseOffsets = spacing * [cos(angles); sin(angles)];
    otherwise
        baseOffsets = zeros(2, max(count, 1));
end
if size(baseOffsets, 2) >= count
    offsets = baseOffsets(:, 1:count);
else
    reps = ceil(count / size(baseOffsets, 2));
    offsets = repmat(baseOffsets, 1, reps);
    offsets = offsets(:, 1:count);
end
end

function value = getField(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end
