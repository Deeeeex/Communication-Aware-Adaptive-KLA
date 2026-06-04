% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_HISTORYCOMPARE - GA history vs w/o history (20 trials)
close all; clc;
scriptDir = resolveScriptDir();
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

%% Trial config
numberOfTrials = 20;
baseSeed = 1;
useFixedSeed = true;

%% Scenario switches
staggeredBirths = true; % true = staggered births, false = all at once
useDistributedFusion = true;
leaderSensor = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis'; % 'Metropolis' or 'Uniform'
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
robustNIS = true;
robustNISMin = 0.3;

%% Sensor configuration
numberOfSensors = 8;
clutterRates = 3 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);

commConfig = struct();
commConfig.level = 2; % default Level 1: bandwidth only
commConfig.globalMaxMeasurementsPerStep = 80;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.maxOutageNodes = 1;

%% Sensor formations (4+4) moving left-to-right (constant velocity)
sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.0;
sensorMotionConfig.initialStates = buildSensorInitialStates();

%% Target formations (3-3-4), all fly toward center
targetFormationConfig = struct();
targetFormationConfig.targetFormationEnabled = true;
targetFormationConfig.targetFormationStaggeredBirths = staggeredBirths;
targetFormationConfig.targetFormationBirthInterval = 8;
targetFormationConfig.targetFormationStartTime = 1;
targetFormationConfig.targetFormationLifeSpan = 100;
targetFormationConfig.targetBirthStates = buildTargetBirthStates();
targetFormationConfig.targetFormationCount = size(targetFormationConfig.targetBirthStates, 2);

%% Allocate trial results
eOspaNoHist = zeros(numberOfTrials, numberOfSensors);
eOspaHist = zeros(numberOfTrials, numberOfSensors);
rmseNoHist = zeros(numberOfTrials, numberOfSensors);
rmseHist = zeros(numberOfTrials, numberOfSensors);
consOspaNoHist = zeros(numberOfTrials, 1);
consOspaHist = zeros(numberOfTrials, 1);
consPosNoHist = zeros(numberOfTrials, 1);
consPosHist = zeros(numberOfTrials, 1);
consCardNoHist = zeros(numberOfTrials, 1);
consCardHist = zeros(numberOfTrials, 1);

%% Trials
for trial = 1:numberOfTrials
    if useFixedSeed
        rng(baseSeed + trial);
    end

    % Generate model with formation scenario
    model = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'GA', 'LBP', 'Formation', ...
        sensorMotionConfig, targetFormationConfig);
    model.sensorCommRange = sensorCommRange;
    model.fusionWeighting = fusionWeighting;
    model.adaptiveFusion = adaptiveFusionConfig;
    model.adaptiveFusion.robustNIS = robustNIS;
    model.adaptiveFusion.robustNISMin = robustNISMin;
    model.sensorFovEnabled = true;
    model.sensorFovHalfAngleDeg = 60;
    model.sensorFovRange = 60000;

    % Generate observations
    [groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
    model.sensorTrajectories = sensorTrajectories;
    [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
    neighborMap = buildNeighborMap4Plus4(numberOfSensors);

    % w/o history
    modelNoHist = model;
    modelNoHist.adaptiveFusion.enabled = true;
    modelNoHist.adaptiveFusion.useNIS = false;
    modelNoHist.adaptiveFusion.useHistory = false;
    [stateEstimatesBySensorNoHist, localModelsNoHist, neighborMap] = runDistributedLmbFilter( ...
        modelNoHist, measurementsDelivered, sensorTrajectories, neighborMap, commStats);

    % history only
    modelHist = model;
    modelHist.adaptiveFusion.enabled = true;
    modelHist.adaptiveFusion.useNIS = false;
    modelHist.adaptiveFusion.useHistory = true;
    [stateEstimatesBySensorHist, localModelsHist, neighborMap] = runDistributedLmbFilter( ...
        modelHist, measurementsDelivered, sensorTrajectories, neighborMap, commStats);

    % Per-sensor metrics
    for s = 1:numberOfSensors
        [eNoHist, ~] = computeSimulationOspa(localModelsNoHist{s}, groundTruthRfs, stateEstimatesBySensorNoHist{s});
        [eHist, ~] = computeSimulationOspa(localModelsHist{s}, groundTruthRfs, stateEstimatesBySensorHist{s});
        eOspaNoHist(trial, s) = mean(eNoHist);
        rmseNoHist(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorNoHist{s}, groundTruthRfs), 'omitnan');
        eOspaHist(trial, s) = mean(eHist);
        rmseHist(trial, s) = mean(computeSetRmseOverTime(stateEstimatesBySensorHist{s}, groundTruthRfs), 'omitnan');
    end

    % Network disagreement metrics
    [posNoHist, cardNoHist, ospaNoHist] = computeConsensusMetrics(stateEstimatesBySensorNoHist, modelNoHist);
    consOspaNoHist(trial) = mean(ospaNoHist);
    consPosNoHist(trial) = mean(posNoHist, 'omitnan');
    consCardNoHist(trial) = mean(cardNoHist);
    [posHist, cardHist, ospaHist] = computeConsensusMetrics(stateEstimatesBySensorHist, modelHist);
    consOspaHist(trial) = mean(ospaHist);
    consPosHist(trial) = mean(posHist, 'omitnan');
    consCardHist(trial) = mean(cardHist);
end

%% Summary
fprintf('=====================================\n');
fprintf('GA Adaptive History Comparison (N=%d)\n', numberOfTrials);
fprintf('useNIS=0, useHistory=0 -> 1\n');
fprintf('=====================================\n');
for s = 1:numberOfSensors
    fprintf('Sensor %d: E-OSPA %.3f -> %.3f, RMSE %.3f -> %.3f\n', ...
        s, mean(eOspaNoHist(:, s)), mean(eOspaHist(:, s)), ...
        mean(rmseNoHist(:, s)), mean(rmseHist(:, s)));
end
fprintf('=====================================\n');
fprintf('Network Disagreement Metrics (w/o history -> history)\n');
fprintf('=====================================\n');
fprintf('OSPA consensus error: %.3f -> %.3f\n', ...
    mean(consOspaNoHist), mean(consOspaHist));
fprintf('Matched localization disagreement: %.3f -> %.3f\n', ...
    mean(consPosNoHist, 'omitnan'), mean(consPosHist, 'omitnan'));
fprintf('Cardinality dispersion: %.3f -> %.3f\n', ...
    mean(consCardNoHist), mean(consCardHist));

%% Write report
reportDir = fullfile(projectRoot, 'RUN', 'GA');
if ~exist(reportDir, 'dir')
    mkdir(reportDir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
reportName = sprintf('GA_HISTORY_COMPARE_%s.md', timestamp);
reportPath = fullfile(reportDir, reportName);
writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
    robustNISMin, sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
    eOspaNoHist, eOspaHist, rmseNoHist, rmseHist, ...
    consOspaNoHist, consOspaHist, ...
    consPosNoHist, consPosHist, ...
    consCardNoHist, consCardHist);
fprintf('Report written: %s\n', reportPath);

%% Local helpers
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

function scriptDir = resolveScriptDir()
    stack = dbstack('-completenames');
    if ~isempty(stack) && isfield(stack, 'file')
        scriptDir = fileparts(stack(1).file);
    else
        scriptDir = pwd;
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
    groupCenters = [70, 80, 70; 80, 0, -80]; % right-up, right, right-down
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
    robustNISMin, sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
    eOspaNoHist, eOspaHist, rmseNoHist, rmseHist, ...
    consOspaNoHist, consOspaHist, ...
    consPosNoHist, consPosHist, ...
    consCardNoHist, consCardHist)

    fid = fopen(reportPath, 'w');
    if fid < 0
        warning('Unable to write report: %s', reportPath);
        return;
    end

    timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    fprintf(fid, '# GA History Comparison (%s)\n\n', timestamp);
    fprintf(fid, 'Comparison order: w/o history -> history\n\n');
    fprintf(fid, '## Run Config\n');
    fprintf(fid, '- Trials: %d\n', numberOfTrials);
    fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
    fprintf(fid, '- useNIS: 0\n');
    fprintf(fid, '- robustNISMin: %.2f\n', robustNISMin);
    fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
    fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
    fprintf(fid, '- useHistory: %d\n', adaptiveFusionConfig.useHistory);
    fprintf(fid, '- historyEmaAlpha: %.2f\n', adaptiveFusionConfig.historyEmaAlpha);
    fprintf(fid, '- historyScale: %.2f\n', adaptiveFusionConfig.historyScale);
    fprintf(fid, '- nisQuantileEnabled: %d\n', adaptiveFusionConfig.nisQuantileEnabled);
    fprintf(fid, '- nisQuantile: %.2f\n', adaptiveFusionConfig.nisQuantile);
    fprintf(fid, '- nisConsistencyConfidence: %.2f\n', adaptiveFusionConfig.nisConsistencyConfidence);
    fprintf(fid, '- nisPenaltyScale: %.2f\n', adaptiveFusionConfig.nisPenaltyScale);
    fprintf(fid, '- nisPenaltyMin: %.2f\n', adaptiveFusionConfig.nisPenaltyMin);
    fprintf(fid, '- nisEmaEnabled: %d\n', adaptiveFusionConfig.nisEmaEnabled);
    fprintf(fid, '- nisEmaAlpha: %.2f\n', adaptiveFusionConfig.nisEmaAlpha);
    fprintf(fid, '- leaderSensor: %d\n\n', leaderSensor);

    fprintf(fid, '## Per-Sensor Metrics (mean across trials)\n');
    fprintf(fid, '| Sensor | E-OSPA (w/o history) | E-OSPA (history) | RMSE (w/o history) | RMSE (history) |\n');
    fprintf(fid, '|:------:|---------------------:|-----------------:|-------------------:|---------------:|\n');
    numberOfSensors = size(eOspaNoHist, 2);
    for s = 1:numberOfSensors
        fprintf(fid, '| %d | %.3f | %.3f | %.3f | %.3f |\n', ...
            s, mean(eOspaNoHist(:, s)), mean(eOspaHist(:, s)), ...
            mean(rmseNoHist(:, s)), mean(rmseHist(:, s)));
    end
    fprintf(fid, '\n');

    fprintf(fid, '## Network Disagreement Metrics (mean across trials)\n');
    fprintf(fid, '- OSPA consensus error: %.3f -> %.3f\n', ...
        mean(consOspaNoHist), mean(consOspaHist));
    fprintf(fid, '- Matched localization disagreement: %.3f -> %.3f\n', ...
        mean(consPosNoHist, 'omitnan'), mean(consPosHist, 'omitnan'));
    fprintf(fid, '- Cardinality dispersion: %.3f -> %.3f\n', ...
        mean(consCardNoHist), mean(consCardHist));

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
