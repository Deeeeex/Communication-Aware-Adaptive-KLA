function [reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkNISCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, adaptiveFusionOverrides, commConfigOverrides, writeReport)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_TIEREDLINKNISCOMPARE
% Compare w/o NIS, robust NIS, and non-robust NIS under tiered per-sensor
% packet-drop rates that remain fixed within each trial.

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
eOspaNo = zeros(numberOfTrials, numberOfSensors);
eOspaRobust = zeros(numberOfTrials, numberOfSensors);
eOspaNis = zeros(numberOfTrials, numberOfSensors);
rmseNo = zeros(numberOfTrials, numberOfSensors);
rmseRobust = zeros(numberOfTrials, numberOfSensors);
rmseNis = zeros(numberOfTrials, numberOfSensors);
consOspaNo = zeros(numberOfTrials, 1);
consOspaRobust = zeros(numberOfTrials, 1);
consOspaNis = zeros(numberOfTrials, 1);
consPosNo = zeros(numberOfTrials, 1);
consPosRobust = zeros(numberOfTrials, 1);
consPosNis = zeros(numberOfTrials, 1);
consCardNo = zeros(numberOfTrials, 1);
consCardRobust = zeros(numberOfTrials, 1);
consCardNis = zeros(numberOfTrials, 1);
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
    model.adaptiveFusion = adaptiveFusionConfig;
    model.sensorFovEnabled = true;
    model.sensorFovHalfAngleDeg = 60;
    model.sensorFovRange = 60000;

    [~, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
    model.sensorTrajectories = sensorTrajectories;
    [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
    pDropBySensorTrials(trial, :) = reshape(commStats.pDropBySensor, 1, []);

    modelNo = model;
    modelNo.adaptiveFusion.enabled = true;
    modelNo.adaptiveFusion.useNIS = false;
    noNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorNo, localModelsNo] = runDistributedLmbFilter( ...
        modelNo, measurementsDelivered, sensorTrajectories, noNeighborMap, commStats);

    modelRobust = model;
    modelRobust.adaptiveFusion.enabled = true;
    modelRobust.adaptiveFusion.useNIS = true;
    modelRobust.adaptiveFusion.robustNIS = true;
    modelRobust.adaptiveFusion.robustNISMin = robustNISMin;
    robustNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorRobust, localModelsRobust] = runDistributedLmbFilter( ...
        modelRobust, measurementsDelivered, sensorTrajectories, robustNeighborMap, commStats);

    modelNis = model;
    modelNis.adaptiveFusion.enabled = true;
    modelNis.adaptiveFusion.useNIS = true;
    modelNis.adaptiveFusion.robustNIS = false;
    nisNeighborMap = buildNeighborMap4Plus4(numberOfSensors);
    [stateEstimatesBySensorNis, localModelsNis] = runDistributedLmbFilter( ...
        modelNis, measurementsDelivered, sensorTrajectories, nisNeighborMap, commStats);

    for s = 1:numberOfSensors
        [eNo, ~] = computeSimulationOspa(localModelsNo{s}, groundTruthRfs, stateEstimatesBySensorNo{s});
        [eRob, ~] = computeSimulationOspa(localModelsRobust{s}, groundTruthRfs, stateEstimatesBySensorRobust{s});
        [eNis, ~] = computeSimulationOspa(localModelsNis{s}, groundTruthRfs, stateEstimatesBySensorNis{s});
        eOspaNo(trial, s) = mean(eNo);
        eOspaRobust(trial, s) = mean(eRob);
        eOspaNis(trial, s) = mean(eNis);
        rmseNo(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorNo{s}, groundTruthRfs), 'omitnan');
        rmseRobust(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorRobust{s}, groundTruthRfs), 'omitnan');
        rmseNis(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorNis{s}, groundTruthRfs), 'omitnan');
    end

    [posNo, cardNo, ospaNo] = computeConsensusMetrics(stateEstimatesBySensorNo, modelNo);
    [posRob, cardRob, ospaRob] = computeConsensusMetrics(stateEstimatesBySensorRobust, modelRobust);
    [posNis, cardNis, ospaNis] = computeConsensusMetrics(stateEstimatesBySensorNis, modelNis);
    consOspaNo(trial) = mean(ospaNo);
    consOspaRobust(trial) = mean(ospaRob);
    consOspaNis(trial) = mean(ospaNis);
    consPosNo(trial) = mean(posNo, 'omitnan');
    consPosRobust(trial) = mean(posRob, 'omitnan');
    consPosNis(trial) = mean(posNis, 'omitnan');
    consCardNo(trial) = mean(cardNo);
    consCardRobust(trial) = mean(cardRob);
    consCardNis(trial) = mean(cardNis);
end

%% Summary
fprintf('=====================================\n');
fprintf('GA Tiered Link NIS Comparison (N=%d)\n', numberOfTrials);
fprintf('Comparison order: w/o NIS -> robust NIS -> NIS\n');
fprintf('Tier levels=%s, counts=%s\n', mat2str(commConfig.pDropLevels, 3), mat2str(commConfig.pDropLevelCounts));
fprintf('=====================================\n');
for s = 1:numberOfSensors
    fprintf('Sensor %d: E-OSPA %.3f -> %.3f -> %.3f, RMSE %.3f -> %.3f -> %.3f, mean pDrop %.3f\n', ...
        s, mean(eOspaNo(:, s)), mean(eOspaRobust(:, s)), mean(eOspaNis(:, s)), ...
        mean(rmseNo(:, s)), mean(rmseRobust(:, s)), mean(rmseNis(:, s)), ...
        mean(pDropBySensorTrials(:, s)));
end
fprintf('=====================================\n');
fprintf('Network Disagreement Metrics (w/o NIS -> robust NIS -> NIS)\n');
fprintf('=====================================\n');
fprintf('OSPA consensus error: %.3f -> %.3f -> %.3f\n', ...
    mean(consOspaNo), mean(consOspaRobust), mean(consOspaNis));
fprintf('Matched localization disagreement: %.3f -> %.3f -> %.3f\n', ...
    mean(consPosNo, 'omitnan'), mean(consPosRobust, 'omitnan'), mean(consPosNis, 'omitnan'));
fprintf('Cardinality dispersion: %.3f -> %.3f -> %.3f\n', ...
    mean(consCardNo), mean(consCardRobust), mean(consCardNis));

summary.consensus.ospaNo = mean(consOspaNo);
summary.consensus.ospaRobust = mean(consOspaRobust);
summary.consensus.ospaNis = mean(consOspaNis);
summary.consensus.posNo = mean(consPosNo, 'omitnan');
summary.consensus.posRobust = mean(consPosRobust, 'omitnan');
summary.consensus.posNis = mean(consPosNis, 'omitnan');
summary.consensus.cardNo = mean(consCardNo);
summary.consensus.cardRobust = mean(consCardRobust);
summary.consensus.cardNis = mean(consCardNis);
summary.consensusTrials.ospa = [consOspaNo, consOspaRobust, consOspaNis];
summary.consensusTrials.pos = [consPosNo, consPosRobust, consPosNis];
summary.consensusTrials.card = [consCardNo, consCardRobust, consCardNis];
summary.trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPath = fullfile(reportDir, sprintf('GA_TIERED_LINK_NIS_COMPARE_N%d_SEED%d_%s.md', ...
        numberOfTrials, baseSeed, timestamp));
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, robustNISMin, ...
        sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, commConfig, ...
        pDropBySensorTrials, eOspaNo, eOspaRobust, eOspaNis, rmseNo, rmseRobust, rmseNis, ...
        consOspaNo, consOspaRobust, consOspaNis, consPosNo, consPosRobust, consPosNis, ...
        consCardNo, consCardRobust, consCardNis);
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

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, robustNISMin, ...
    sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, commConfig, ...
    pDropBySensorTrials, eOspaNo, eOspaRobust, eOspaNis, rmseNo, rmseRobust, rmseNis, ...
    consOspaNo, consOspaRobust, consOspaNis, consPosNo, consPosRobust, consPosNis, ...
    consCardNo, consCardRobust, consCardNis)
fid = fopen(reportPath, 'w');
if fid < 0
    return;
end
fprintf(fid, '# GA Tiered Link NIS Comparison (%s)\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Comparison order: w/o NIS -> robust NIS -> NIS\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
if useFixedSeed
    fprintf(fid, '- trialSeeds: %s\n', mat2str(computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)));
end
fprintf(fid, '- robustNISMin: %.2f\n', robustNISMin);
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- pDropLevels: %s\n', mat2str(getField(commConfig, 'pDropLevels', []), 3));
fprintf(fid, '- pDropLevelCounts: %s\n\n', mat2str(getField(commConfig, 'pDropLevelCounts', [])));
fprintf(fid, '## Per-Trial pDropBySensor\n');
for trial = 1:size(pDropBySensorTrials, 1)
    fprintf(fid, '- Trial %d: %s\n', trial, mat2str(pDropBySensorTrials(trial, :), 4));
end
fprintf(fid, '\n## Per-Trial Network Disagreement Metrics\n');
fprintf(fid, '| Trial | Seed | Arm | OSPA | RMSE | Cardinality |\n');
fprintf(fid, '|------:|-----:|:----|-----:|-----:|------------:|\n');
trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
armNames = {'w/o NIS', 'robust NIS', 'plain NIS'};
for trial = 1:numberOfTrials
    fprintf(fid, '| %d | %.0f | %s | %.6f | %.6f | %.6f |\n', trial, trialSeeds(trial), armNames{1}, consOspaNo(trial), consPosNo(trial), consCardNo(trial));
    fprintf(fid, '| %d | %.0f | %s | %.6f | %.6f | %.6f |\n', trial, trialSeeds(trial), armNames{2}, consOspaRobust(trial), consPosRobust(trial), consCardRobust(trial));
    fprintf(fid, '| %d | %.0f | %s | %.6f | %.6f | %.6f |\n', trial, trialSeeds(trial), armNames{3}, consOspaNis(trial), consPosNis(trial), consCardNis(trial));
end
fprintf(fid, '\n## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '- OSPA consensus error: %.3f -> %.3f -> %.3f\n', mean(consOspaNo), mean(consOspaRobust), mean(consOspaNis));
fprintf(fid, '- Matched localization disagreement: %.3f -> %.3f -> %.3f\n', mean(consPosNo, 'omitnan'), mean(consPosRobust, 'omitnan'), mean(consPosNis, 'omitnan'));
fprintf(fid, '- Cardinality dispersion: %.3f -> %.3f -> %.3f\n', mean(consCardNo), mean(consCardRobust), mean(consCardNis));
fprintf(fid, '\n## Network Disagreement Metrics With Trial Variability\n');
writeMetricStatsTable(fid, armNames, {'OSPA', 'RMSE', 'Cardinality'}, ...
    { [consOspaNo, consOspaRobust, consOspaNis], [consPosNo, consPosRobust, consPosNis], [consCardNo, consCardRobust, consCardNis] }, ...
    [false, true, false]);
fclose(fid);
end

function seeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)
if useFixedSeed
    seeds = baseSeed + (1:numberOfTrials);
else
    seeds = NaN(1, numberOfTrials);
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
