function imagePath = plotStandardIdealScenarioComparison(seed, outputPath)
% PLOTSTANDARDIDEALSCENARIOCOMPARISON
% Create a static sensor-1 comparison figure for the standard ideal fixed
% scenario, styled after the paper's time-varying x/y track plot.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

if nargin < 1 || isempty(seed)
    seed = 2;
end
if nargin < 2 || isempty(outputPath)
    outputPath = fullfile(projectRoot, 'RUN', 'IDEAL', ...
        sprintf('STANDARD_FIXED_IDEAL_SENSOR1_COMPARE_%d.png', seed));
end

rng(seed);

numberOfSensors = 4;
clutterRates = 5 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);
scenarioType = 'Fixed';

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

baseModel = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'GA', 'LBP', scenarioType);
baseModel.simulationLength = 100;
baseModel.gaSensorWeights = ones(1, numberOfSensors) / numberOfSensors;
baseModel.aaSensorWeights = ones(1, numberOfSensors) / numberOfSensors;
baseModel.fusionWeighting = 'Uniform';
baseModel.adaptiveFusion = struct('enabled', false);

[groundTruth, measurements, ~] = generateMultisensorGroundTruth(baseModel);
[measurementsDelivered, commStats] = applyCommunicationModel(measurements, baseModel, commConfig);

modelGa = baseModel;
modelGa.lmbParallelUpdateMode = 'GA';
stateGa = runParallelUpdateLmbFilter(modelGa, measurementsDelivered, commStats, []);

modelBest = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'GA', 'LBP', scenarioType);
modelBest.simulationLength = 100;
modelBest.gaSensorWeights = baseModel.gaSensorWeights;
modelBest.aaSensorWeights = baseModel.aaSensorWeights;
modelBest.fusionWeighting = 'Uniform';
modelBest.adaptiveFusion = adaptiveFusionConfig;
stateBest = runParallelUpdateLmbFilter(modelBest, measurementsDelivered, commStats, []);

sensorIdx = 1;
[measurementTimes, measX, measY] = packSensorMeasurements(measurementsDelivered, sensorIdx, baseModel.T);
[truthSegmentsX, truthSegmentsY] = packGroundTruthSegments(groundTruth);
[estTimesGa, estXGa, estYGa] = packStateEstimates(stateGa, baseModel.T);
[estTimesBest, estXBest, estYBest] = packStateEstimates(stateBest, baseModel.T);

fig = figure('Visible', 'off', 'Position', [120, 120, 1200, 720], 'Color', 'w');
ax1 = subplot(2, 2, 1, 'Parent', fig);
ax2 = subplot(2, 2, 2, 'Parent', fig);
ax3 = subplot(2, 2, 3, 'Parent', fig);
ax4 = subplot(2, 2, 4, 'Parent', fig);

renderAxis(ax1, truthSegmentsX, measurementTimes, measX, estTimesGa, estXGa, ...
    'Fixed GA: x-coordinate vs time', 'x (m)', [0.15, 0.45, 0.75]);
renderAxis(ax2, truthSegmentsY, measurementTimes, measY, estTimesGa, estYGa, ...
    'Fixed GA: y-coordinate vs time', 'y (m)', [0.15, 0.45, 0.75]);
renderAxis(ax3, truthSegmentsX, measurementTimes, measX, estTimesBest, estXBest, ...
    'Current Best: x-coordinate vs time', 'x (m)', [0.80, 0.33, 0.10]);
renderAxis(ax4, truthSegmentsY, measurementTimes, measY, estTimesBest, estYBest, ...
    'Current Best: y-coordinate vs time', 'y (m)', [0.80, 0.33, 0.10]);
annotation(fig, 'textbox', [0.20 0.955 0.60 0.04], 'String', ...
    sprintf('Standard Ideal Fixed Scenario, Sensor %d, seed=%d', sensorIdx, seed), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
print(fig, outputPath, '-dpng', '-r180');
close(fig);
imagePath = outputPath;
fprintf('Saved static comparison figure: %s\n', imagePath);
end

function renderAxis(ax, truthSegments, measurementTimes, measurementValues, estTimes, estValues, plotTitle, yLabelText, estimateColor)
axes(ax); %#ok<LAXES>
hold(ax, 'on');
box(ax, 'on');
grid(ax, 'on');

for idx = 1:numel(truthSegments)
    seg = truthSegments{idx};
    plot(ax, seg(1, :), seg(2, :), 'k-', 'LineWidth', 2.0);
end
plot(ax, measurementTimes, measurementValues, '.', 'Color', [0.72, 0.72, 0.72], 'MarkerSize', 6);
plot(ax, estTimes, estValues, 'o', 'Color', estimateColor, ...
    'MarkerFaceColor', estimateColor, 'MarkerSize', 3);

xlabel(ax, 'Time (s)');
ylabel(ax, yLabelText);
title(ax, plotTitle);
legend(ax, {'True tracks', 'Measurements', 'Estimates'}, 'Location', 'northeast');
end

function [times, xValues, yValues] = packSensorMeasurements(measurements, sensorIdx, dt)
numSteps = size(measurements, 2);
times = zeros(1, 0);
xValues = zeros(1, 0);
yValues = zeros(1, 0);
for t = 1:numSteps
    entries = measurements{sensorIdx, t};
    if isempty(entries)
        continue;
    end
    Z = [entries{:}];
    count = size(Z, 2);
    times = [times, dt * (t - 1) * ones(1, count)]; %#ok<AGROW>
    xValues = [xValues, Z(1, :)]; %#ok<AGROW>
    yValues = [yValues, Z(2, :)]; %#ok<AGROW>
end
end

function [segmentsX, segmentsY] = packGroundTruthSegments(groundTruth)
segmentsX = cell(1, numel(groundTruth));
segmentsY = cell(1, numel(groundTruth));
for idx = 1:numel(groundTruth)
    traj = groundTruth{idx};
    times = traj(1, :);
    segmentsX{idx} = [times; traj(2, :)];
    segmentsY{idx} = [times; traj(3, :)];
end
end

function [times, xValues, yValues] = packStateEstimates(stateEstimates, dt)
numSteps = numel(stateEstimates.mu);
times = zeros(1, 0);
xValues = zeros(1, 0);
yValues = zeros(1, 0);
for t = 1:numSteps
    if isempty(stateEstimates.mu{t})
        continue;
    end
    X = [stateEstimates.mu{t}{:}];
    count = size(X, 2);
    times = [times, dt * (t - 1) * ones(1, count)]; %#ok<AGROW>
    xValues = [xValues, X(1, :)]; %#ok<AGROW>
    yValues = [yValues, X(2, :)]; %#ok<AGROW>
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
