% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_AA - 4+4 sensor formations + formation targets demo (AA)
close all; clc;
scriptDir = resolveScriptDir();
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

%% Switch: staggered target births
staggeredBirths = true; % true = staggered births, false = all at once
makeGif = true;
outputDir = fullfile(projectRoot, 'RUN', 'AA');
gifPath = fullfile(outputDir, 'formation_animation_4plus4_AA.gif');
frameSkip = 1;
frameDelay = 0.08;
fovHalfAngleDeg = 60;
fovRange = 60000;
fovColor = [0.7 0.7 0.7];
useDistributedFusion = true;
leaderSensor = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis'; % 'Metropolis' or 'Uniform'
compareAdaptiveWeights = true;
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
    'nisPenaltyUpperScale', 4.0, ...
    'nisPenaltyLowerPower', 2.0, ...
    'nisPenaltyUpperPower', 2.0, ...
    'nisEmaEnabled', true, ...
    'nisEmaAlpha', 0.7);
gifPathBase = fullfile(outputDir, 'formation_animation_4plus4_AA_base.gif');
gifPathAdaptive = fullfile(outputDir, 'formation_animation_4plus4_AA_adaptive.gif');
%% Sensor configuration
numberOfSensors = 8;
clutterRates = 3 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);

commConfig = struct();
commConfig.level = 1; % default Level 1: bandwidth only
commConfig.globalMaxMeasurementsPerStep = 90;
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

%% Generate model with formation scenario
model = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'AA', 'LBP', 'Formation', ...
    sensorMotionConfig, targetFormationConfig);
model.sensorCommRange = sensorCommRange;
model.fusionWeighting = fusionWeighting;
model.adaptiveFusion = adaptiveFusionConfig;
model.sensorFovEnabled = true;
model.sensorFovHalfAngleDeg = fovHalfAngleDeg;
model.sensorFovRange = fovRange;

%% Generate observations
[groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
model.sensorTrajectories = sensorTrajectories;
% Apply communication model (Level 1 by default)
[measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
neighborMap = buildNeighborMap4Plus4(numberOfSensors);

%% Run filter (distributed local fusion)
stateEstimatesBySensor = [];
localModels = [];
stateEstimatesBase = [];
stateEstimatesAdaptive = [];
if useDistributedFusion
    if compareAdaptiveWeights
        modelBase = model;
        modelBase.adaptiveFusion.enabled = false;
        [stateEstimatesBySensorBase, localModelsBase, neighborMap] = runDistributedLmbFilter( ...
            modelBase, measurementsDelivered, sensorTrajectories, neighborMap, commStats);

        modelAdaptive = model;
        modelAdaptive.adaptiveFusion.enabled = true;
        [stateEstimatesBySensorAdaptive, localModelsAdaptive, neighborMap] = runDistributedLmbFilter( ...
            modelAdaptive, measurementsDelivered, sensorTrajectories, neighborMap, commStats);

        stateEstimatesBySensor = stateEstimatesBySensorAdaptive;
        localModels = localModelsAdaptive;
        stateEstimatesBase = stateEstimatesBySensorBase{leaderSensor};
        stateEstimatesAdaptive = stateEstimatesBySensorAdaptive{leaderSensor};
        stateEstimates = stateEstimatesAdaptive;
    else
        [stateEstimatesBySensor, localModels, neighborMap] = runDistributedLmbFilter( ...
            model, measurementsDelivered, sensorTrajectories, neighborMap, commStats);
        stateEstimates = stateEstimatesBySensor{leaderSensor};
    end
else
    if compareAdaptiveWeights
        modelBase = model;
        modelBase.adaptiveFusion.enabled = false;
        stateEstimatesBase = runParallelUpdateLmbFilter(modelBase, measurementsDelivered, commStats, sensorTrajectories);

        modelAdaptive = model;
        modelAdaptive.adaptiveFusion.enabled = true;
        stateEstimatesAdaptive = runParallelUpdateLmbFilter(modelAdaptive, measurementsDelivered, commStats, sensorTrajectories);
        stateEstimates = stateEstimatesAdaptive;
    else
        stateEstimates = runParallelUpdateLmbFilter(model, measurementsDelivered, commStats, sensorTrajectories);
    end
end

%% Compute OSPA
[eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
fprintf('Average E-OSPA: %.3f\n', mean(eOspa));
fprintf('Average H-OSPA: %.3f\n', mean(hOspa));
if useDistributedFusion
    fprintf('=====================================\n');
    fprintf('Per-Sensor OSPA (local fusion)\n');
    fprintf('=====================================\n');
    if compareAdaptiveWeights
        for s = 1:numberOfSensors
            [eBase, ~] = computeSimulationOspa(localModelsBase{s}, groundTruthRfs, stateEstimatesBySensorBase{s});
            [eAda, ~] = computeSimulationOspa(localModelsAdaptive{s}, groundTruthRfs, stateEstimatesBySensorAdaptive{s});
            rmseBase = computeSetRmseOverTime(stateEstimatesBySensorBase{s}, groundTruthRfs);
            rmseAda = computeSetRmseOverTime(stateEstimatesBySensorAdaptive{s}, groundTruthRfs);
            fprintf('Sensor %d: E-OSPA %.3f -> %.3f (%.3f), RMSE %.3f -> %.3f (%.3f), neighbors=%s\n', ...
                s, mean(eBase), mean(eAda), mean(eAda) - mean(eBase), ...
                mean(rmseBase, 'omitnan'), mean(rmseAda, 'omitnan'), ...
                mean(rmseAda, 'omitnan') - mean(rmseBase, 'omitnan'), mat2str(neighborMap{s}));
        end
    else
        for s = 1:numberOfSensors
            [eOspaLocal, ~] = computeSimulationOspa(localModels{s}, groundTruthRfs, stateEstimatesBySensor{s});
            rmseLocal = computeSetRmseOverTime(stateEstimatesBySensor{s}, groundTruthRfs);
            fprintf('Sensor %d: E-OSPA=%.3f, RMSE=%.3f, neighbors=%s\n', ...
                s, mean(eOspaLocal), mean(rmseLocal, 'omitnan'), mat2str(neighborMap{s}));
        end
    end
end

%% Network disagreement metrics (distributed only)
if useDistributedFusion
    if compareAdaptiveWeights
        [posBase, cardBase, ospaBase] = computeConsensusMetrics(stateEstimatesBySensorBase, modelBase);
        [posAda, cardAda, ospaAda] = computeConsensusMetrics(stateEstimatesBySensorAdaptive, modelAdaptive);
        fprintf('=====================================\n');
        fprintf('Network Disagreement Metrics (base -> adaptive)\n');
        fprintf('=====================================\n');
        fprintf('OSPA consensus error: %.3f -> %.3f (%.3f)\n', mean(ospaBase), mean(ospaAda), mean(ospaAda) - mean(ospaBase));
        fprintf('Matched localization disagreement: %.3f -> %.3f (%.3f)\n', mean(posBase, 'omitnan'), mean(posAda, 'omitnan'), mean(posAda, 'omitnan') - mean(posBase, 'omitnan'));
        fprintf('Cardinality dispersion: %.3f -> %.3f (%.3f)\n', mean(cardBase), mean(cardAda), mean(cardAda) - mean(cardBase));
        plotConsensusMetrics(posBase, cardBase, ospaBase, posAda, cardAda, ospaAda);
    else
        [posCons, cardCons, ospaCons] = computeConsensusMetrics(stateEstimatesBySensor, model);
        fprintf('=====================================\n');
        fprintf('Network Disagreement Metrics\n');
        fprintf('=====================================\n');
        fprintf('OSPA consensus error: %.3f\n', mean(ospaCons));
        fprintf('Matched localization disagreement: %.3f\n', mean(posCons, 'omitnan'));
        fprintf('Cardinality dispersion: %.3f\n', mean(cardCons));
        plotConsensusMetrics(posCons, cardCons, ospaCons, [], [], []);
    end
end

%% Plotting
if useDistributedFusion
    measurementsLeader = measurementsDelivered;
    for s = 1:numberOfSensors
        if ~ismember(s, neighborMap{leaderSensor})
            for t = 1:size(measurementsLeader, 2)
                measurementsLeader{s, t} = {};
            end
        end
    end
    plotMultisensorResults(model, measurementsLeader, groundTruth, stateEstimates, groundTruthRfs);
else
    plotMultisensorResults(model, measurementsDelivered, groundTruth, stateEstimates, groundTruthRfs);
end

%% Local helpers
if makeGif
    ensureGifDir(gifPath);
    ensureGifDir(gifPathBase);
    ensureGifDir(gifPathAdaptive);
    if compareAdaptiveWeights && ~isempty(stateEstimatesBase) && ~isempty(stateEstimatesAdaptive)
        animateScenario(model, groundTruth, stateEstimatesBase, sensorTrajectories, gifPathBase, frameSkip, frameDelay, fovHalfAngleDeg, fovRange, fovColor, leaderSensor);
        animateScenario(model, groundTruth, stateEstimatesAdaptive, sensorTrajectories, gifPathAdaptive, frameSkip, frameDelay, fovHalfAngleDeg, fovRange, fovColor, leaderSensor);
        fprintf('Saved animation (base) to %s\n', gifPathBase);
        fprintf('Saved animation (adaptive) to %s\n', gifPathAdaptive);
    else
        animateScenario(model, groundTruth, stateEstimates, sensorTrajectories, gifPath, frameSkip, frameDelay, fovHalfAngleDeg, fovRange, fovColor, leaderSensor);
        fprintf('Saved animation to %s\n', gifPath);
    end
end

function y = smooth1d(x, windowSize)
    if nargin < 2
        windowSize = 5;
    end
    if numel(x) < 3 || windowSize < 2
        y = x;
        return;
    end
    windowSize = min(windowSize, numel(x));
    w = ones(1, windowSize) / windowSize;
    y = conv(x, w, 'same');
end

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

function ensureGifDir(pathStr)
    if nargin < 1 || isempty(pathStr)
        return;
    end
    outDir = fileparts(pathStr);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
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

function animateScenario(model, groundTruth, stateEstimates, sensorTrajectories, gifPath, frameSkip, frameDelay, fovHalfAngleDeg, fovRange, fovColor, leaderSensor)
    outDir = fileparts(gifPath);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    simLength = numel(stateEstimates.mu);
    if nargin < 11 || isempty(leaderSensor)
        leaderSensor = 1;
    end
    fig = figure('Position', [150, 150, 900, 600], 'Visible', 'off', 'IntegerHandle', 'off');
    set(fig, 'DefaultLegendAutoUpdate', 'off');
    ax = axes('Parent', fig);
    for t = 1:frameSkip:simLength
        cla(ax);
        axes(ax);
        hold on; grid on; axis equal;
        legend('off');
        xlim(ax, model.observationSpaceLimits(1, :));
        ylim(ax, model.observationSpaceLimits(2, :));
        title(ax, sprintf('t = %d', t));
        xlabel(ax, 'X Position');
        ylabel(ax, 'Y Position');

        % Ground truth trajectories up to t
        for i = 1:length(groundTruth)
            traj = groundTruth{i};
            idx = traj(1, :) <= t;
            if any(idx)
                plot(ax, traj(2, idx), traj(3, idx), 'b-', 'LineWidth', 1.0);
            end
        end

        % Estimated trajectories up to t
        for i = 1:length(stateEstimates.objects)
            obj = stateEstimates.objects(i);
            if obj.trajectoryLength > 0
                idx = obj.timestamps(1:obj.trajectoryLength) <= t;
                if any(idx)
                    plot(ax, obj.trajectory(1, idx), obj.trajectory(2, idx), '-', ...
                        'Color', [0.85 0.2 0.2], 'LineWidth', 1.0);
                end
            end
        end

        % Sensor trajectories up to t
        for s = 1:model.numberOfSensors
            sensorTraj = sensorTrajectories{s};
            plot(ax, sensorTraj(1, 1:t), sensorTraj(2, 1:t), 'k--', 'LineWidth', 1.2);
            if s == leaderSensor
                plot(ax, sensorTraj(1, t), sensorTraj(2, t), 's', 'MarkerSize', 9, 'MarkerFaceColor', [0.9 0.2 0.2], 'Color', [0.6 0 0]);
            else
                plot(ax, sensorTraj(1, t), sensorTraj(2, t), 'k^', 'MarkerSize', 7, 'MarkerFaceColor', 'k');
            end
            plotFovRays(sensorTraj(1:2, t), sensorTraj(3:4, t), fovHalfAngleDeg, fovRange, fovColor);
        end

        % Ground truth targets at t
        truthPos = collectTruthAtTime(groundTruth, t);
        if ~isempty(truthPos)
            plot(ax, truthPos(1, :), truthPos(2, :), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
        end

        % Estimates at t
        estPos = collectEstimatesAtTime(stateEstimates, t);
        if ~isempty(estPos)
            plot(ax, estPos(1, :), estPos(2, :), 'rx', 'MarkerSize', 7, 'LineWidth', 1.5);
        end

        drawnow;
        frame = getframe(fig);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        if t == 1
            imwrite(imind, cm, gifPath, 'gif', 'Loopcount', inf, 'DelayTime', frameDelay);
        else
            imwrite(imind, cm, gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', frameDelay);
        end
    end
    close(fig);
    fprintf('Saved animation to %s\n', gifPath);
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

function plotConsensusMetrics(posBase, cardBase, ospaBase, posAda, cardAda, ospaAda)
    figure('Position', [200, 200, 1350, 400]);
    subplot(1, 3, 1);
    hold on; grid on;
    plot(ospaBase, 'LineWidth', 1.5, 'Color', [0.2 0.2 0.8]);
    if ~isempty(ospaAda)
        plot(ospaAda, 'LineWidth', 1.5, 'Color', [0.85 0.2 0.2]);
        legend('Base', 'Adaptive', 'Location', 'best');
    else
        legend('Consensus', 'Location', 'best');
    end
    title('OSPA Consensus Error');
    xlabel('Time Step');
    ylabel('OSPA');

    subplot(1, 3, 2);
    hold on; grid on;
    plot(posBase, 'LineWidth', 1.5, 'Color', [0.2 0.2 0.8]);
    if ~isempty(posAda)
        plot(posAda, 'LineWidth', 1.5, 'Color', [0.85 0.2 0.2]);
        legend('Base', 'Adaptive', 'Location', 'best');
    else
        legend('Consensus', 'Location', 'best');
    end
    title('Matched Localization Disagreement');
    xlabel('Time Step');
    ylabel('RMSE');

    subplot(1, 3, 3);
    hold on; grid on;
    plot(cardBase, 'LineWidth', 1.5, 'Color', [0.2 0.2 0.8]);
    if ~isempty(cardAda)
        plot(cardAda, 'LineWidth', 1.5, 'Color', [0.85 0.2 0.2]);
        legend('Base', 'Adaptive', 'Location', 'best');
    else
        legend('Consensus', 'Location', 'best');
    end
    title('Cardinality Dispersion (MAD)');
    xlabel('Time Step');
    ylabel('Mean Abs Deviation');
end

function truthPos = collectTruthAtTime(groundTruth, t)
    truthPos = [];
    for i = 1:length(groundTruth)
        traj = groundTruth{i};
        idx = find(traj(1, :) == t, 1, 'first');
        if ~isempty(idx)
            truthPos(:, end+1) = traj(2:3, idx); %#ok<AGROW>
        end
    end
end

function estPos = collectEstimatesAtTime(stateEstimates, t)
    estPos = [];
    if t <= numel(stateEstimates.mu)
        mu = stateEstimates.mu{t};
        if ~isempty(mu)
            for k = 1:numel(mu)
                estPos(:, end+1) = mu{k}(1:2); %#ok<AGROW>
            end
        end
    end
end

function plotFovRays(pos, vel, halfAngleDeg, range, color)
    if nargin < 5
        color = [0.7 0.7 0.7];
    end
    if nargin < 4
        range = 60;
    end
    if nargin < 3
        halfAngleDeg = 60;
    end
    if isempty(pos) || isempty(vel)
        return;
    end
    v = vel(:);
    if norm(v) < 1e-6
        return;
    end
    v = v / norm(v);
    baseAngle = atan2(v(2), v(1));
    a1 = baseAngle + deg2rad(halfAngleDeg);
    a2 = baseAngle - deg2rad(halfAngleDeg);
    p1 = pos + range * [cos(a1); sin(a1)];
    p2 = pos + range * [cos(a2); sin(a2)];
    line([pos(1) p1(1)], [pos(2) p1(2)], 'LineStyle', '--', 'Color', color, 'LineWidth', 1);
    line([pos(1) p2(1)], [pos(2) p2(2)], 'LineStyle', '--', 'Color', color, 'LineWidth', 1);
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
