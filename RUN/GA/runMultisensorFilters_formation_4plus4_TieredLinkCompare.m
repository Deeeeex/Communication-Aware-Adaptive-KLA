function [reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, adaptiveFusionOverrides, commConfigOverrides, writeReport)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_TIEREDLINKCOMPARE
% Compare fixed fusion weights against adaptive robust-NIS weights under
% tiered per-sensor packet-drop rates that remain fixed within each trial.

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
adaptiveFusionConfig = struct( ...
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
    'nisEmaAlpha', 0.7, ...
    'useNIS', true, ...
    'robustNIS', true, ...
    'robustNISMin', 0.3);

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

adaptiveFusionConfig = mergeStructFields(adaptiveFusionConfig, adaptiveFusionOverrides);
commConfig = mergeStructFields(commConfig, commConfigOverrides);

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
eOspaBase = zeros(numberOfTrials, numberOfSensors);
eOspaAdaptive = zeros(numberOfTrials, numberOfSensors);
rmseBase = zeros(numberOfTrials, numberOfSensors);
rmseAdaptive = zeros(numberOfTrials, numberOfSensors);
consOspaBase = zeros(numberOfTrials, 1);
consOspaAdaptive = zeros(numberOfTrials, 1);
consPosBase = zeros(numberOfTrials, 1);
consPosAdaptive = zeros(numberOfTrials, 1);
consCardBase = zeros(numberOfTrials, 1);
consCardAdaptive = zeros(numberOfTrials, 1);
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
        [eBase, ~] = computeSimulationOspa(localModelsBase{s}, groundTruthRfs, stateEstimatesBySensorBase{s});
        [eAdaptive, ~] = computeSimulationOspa(localModelsAdaptive{s}, groundTruthRfs, stateEstimatesBySensorAdaptive{s});
        eOspaBase(trial, s) = mean(eBase);
        eOspaAdaptive(trial, s) = mean(eAdaptive);
        rmseBase(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorBase{s}, groundTruthRfs), 'omitnan');
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

%% Summary
fprintf('=====================================\n');
fprintf('GA Tiered Link Comparison (N=%d)\n', numberOfTrials);
fprintf('Control=fixed weights, Experiment=adaptive robust NIS\n');
fprintf('Tier levels=%s, counts=%s\n', mat2str(commConfig.pDropLevels, 3), mat2str(commConfig.pDropLevelCounts));
fprintf('=====================================\n');
for s = 1:numberOfSensors
    fprintf('Sensor %d: E-OSPA %.3f -> %.3f, RMSE %.3f -> %.3f, mean pDrop %.3f\n', ...
        s, mean(eOspaBase(:, s)), mean(eOspaAdaptive(:, s)), ...
        mean(rmseBase(:, s)), mean(rmseAdaptive(:, s)), mean(pDropBySensorTrials(:, s)));
end
fprintf('=====================================\n');
fprintf('Network Disagreement Metrics (fixed weights -> adaptive robust NIS)\n');
fprintf('=====================================\n');
fprintf('OSPA consensus error: %.3f -> %.3f\n', ...
    mean(consOspaBase), mean(consOspaAdaptive));
fprintf('Matched localization disagreement: %.3f -> %.3f\n', ...
    mean(consPosBase, 'omitnan'), mean(consPosAdaptive, 'omitnan'));
fprintf('Cardinality dispersion: %.3f -> %.3f\n', ...
    mean(consCardBase), mean(consCardAdaptive));

summary.local.eOspaBase = mean(eOspaBase, 1);
summary.local.eOspaAdaptive = mean(eOspaAdaptive, 1);
summary.local.rmseBase = mean(rmseBase, 1);
summary.local.rmseAdaptive = mean(rmseAdaptive, 1);
summary.consensus.ospaBase = mean(consOspaBase);
summary.consensus.ospaAdaptive = mean(consOspaAdaptive);
summary.consensus.posBase = mean(consPosBase, 'omitnan');
summary.consensus.posAdaptive = mean(consPosAdaptive, 'omitnan');
summary.consensus.cardBase = mean(consCardBase);
summary.consensus.cardAdaptive = mean(consCardAdaptive);
summary.delta.ospa = summary.consensus.ospaAdaptive - summary.consensus.ospaBase;
summary.delta.pos = summary.consensus.posAdaptive - summary.consensus.posBase;
summary.delta.card = summary.consensus.cardAdaptive - summary.consensus.cardBase;
summary.pDropBySensorTrials = pDropBySensorTrials;
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);
summary.commConfig = commConfig;
summary.adaptiveFusionConfig = adaptiveFusionConfig;

%% Write report
if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportName = sprintf('GA_TIERED_LINK_COMPARE_%s.md', timestamp);
    reportPath = fullfile(reportDir, reportName);
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
        sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
        commConfig, pDropBySensorTrials, eOspaBase, eOspaAdaptive, rmseBase, rmseAdaptive, ...
        consOspaBase, consOspaAdaptive, consPosBase, consPosAdaptive, consCardBase, consCardAdaptive);
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

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
    sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
    commConfig, pDropBySensorTrials, eOspaBase, eOspaAdaptive, rmseBase, rmseAdaptive, ...
    consOspaBase, consOspaAdaptive, consPosBase, consPosAdaptive, consCardBase, consCardAdaptive)

fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf(fid, '# GA Tiered Link Comparison (%s)\n\n', timestamp);
fprintf(fid, 'Comparison order: fixed weights -> adaptive robust NIS\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- linkModel: %s\n', getField(commConfig, 'linkModel', 'fixed'));
fprintf(fid, '- pDrop target mean: %.3f\n', getField(commConfig, 'pDrop', 0));
fprintf(fid, '- pDropLevels: %s\n', mat2str(getField(commConfig, 'pDropLevels', []), 3));
fprintf(fid, '- pDropLevelCounts: %s\n', mat2str(getField(commConfig, 'pDropLevelCounts', [])));
fprintf(fid, '- adaptive useNIS: %d\n', adaptiveFusionConfig.useNIS);
fprintf(fid, '- adaptive robustNIS: %d\n', adaptiveFusionConfig.robustNIS);
fprintf(fid, '- adaptive robustNISMin: %.2f\n', adaptiveFusionConfig.robustNISMin);
fprintf(fid, '- adaptive useHistory: %d\n\n', adaptiveFusionConfig.useHistory);

fprintf(fid, '## Per-Trial pDropBySensor\n');
for trial = 1:size(pDropBySensorTrials, 1)
    fprintf(fid, '- Trial %d: %s\n', trial, mat2str(pDropBySensorTrials(trial, :), 4));
end
fprintf(fid, '\n');

fprintf(fid, '## Mean pDropBySensor Across Trials\n');
fprintf(fid, '- %s\n\n', mat2str(mean(pDropBySensorTrials, 1), 4));

fprintf(fid, '## Per-Sensor Metrics (mean across trials)\n');
fprintf(fid, '| Sensor | Mean pDrop | E-OSPA (fixed) | E-OSPA (adaptive) | RMSE (fixed) | RMSE (adaptive) |\n');
fprintf(fid, '|:------:|-----------:|---------------:|------------------:|-------------:|----------------:|\n');
numberOfSensors = size(eOspaBase, 2);
for s = 1:numberOfSensors
    fprintf(fid, '| %d | %.3f | %.3f | %.3f | %.3f | %.3f |\n', ...
        s, mean(pDropBySensorTrials(:, s)), mean(eOspaBase(:, s)), mean(eOspaAdaptive(:, s)), ...
        mean(rmseBase(:, s)), mean(rmseAdaptive(:, s)));
end
fprintf(fid, '\n');

fprintf(fid, '## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '- OSPA consensus error: %.3f -> %.3f\n', ...
    mean(consOspaBase), mean(consOspaAdaptive));
fprintf(fid, '- Matched localization disagreement: %.3f -> %.3f\n', ...
    mean(consPosBase, 'omitnan'), mean(consPosAdaptive, 'omitnan'));
fprintf(fid, '- Cardinality dispersion: %.3f -> %.3f\n', ...
    mean(consCardBase), mean(consCardAdaptive));

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
    if pairCount > 0
        posConsensus(t) = pairSum / pairCount;
    else
        posConsensus(t) = NaN;
    end
    if ospaCount > 0
        ospaConsensus(t) = ospaSum / ospaCount;
    else
        ospaConsensus(t) = NaN;
    end
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
if isempty(matched)
    dist = NaN;
    return;
end
dist = sqrt(mean(matched.^2));
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
    return;
end
rmse = sqrt(mean(matched.^2));
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
