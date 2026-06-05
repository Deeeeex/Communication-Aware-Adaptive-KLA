function [reportPath, summary] = runMultisensorFilters_formation_4plus4_CommLevelThreeMethodCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, outputCsvPath)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_COMMLEVELTHREEMETHODCOMPARE
% Compare Fixed Metropolis, Balanced mode, and Cardinality-critical mode
% across communication levels 0-3.
%
% This runner is intended for the communication-robustness figure/table. It
% reuses the main fidFiaExistenceRefinement arm set and selects only:
%   Fixed Metropolis, Balanced mode, Cardinality-critical mode.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
addpath(fullfile(projectRoot, 'RUN', 'GA'));
setPath;

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 50;
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

reportDir = fullfile(projectRoot, 'RUN', 'GA');
if nargin < 5 || isempty(outputCsvPath)
    outputCsvPath = fullfile(reportDir, sprintf( ...
        'GA_COMM_LEVEL_THREE_METHOD_N%d_SEED%d_latest.csv', numberOfTrials, baseSeed));
end

levels = 0:3;
levelLabels = {'none', 'bandwidth cap', 'tiered link loss', 'node outage'};
methodDisplayNames = {'Fixed Metropolis', 'Balanced mode', 'Cardinality-critical mode'};
armSelection = [1 4 5];

summary = struct();
summary.levels = levels;
summary.levelLabels = levelLabels;
summary.methodDisplayNames = methodDisplayNames;
summary.numberOfTrials = numberOfTrials;
summary.baseSeed = baseSeed;
summary.useFixedSeed = useFixedSeed;
summary.outputCsvPath = outputCsvPath;
summary.levelSummaries = cell(1, numel(levels));
summary.levelReportPaths = cell(1, numel(levels));

networkMetricFields = {'ospa', 'pos', 'card'};
networkMetricNames = {'OSPA consensus error', 'Matched localization disagreement', 'Cardinality dispersion'};
localMetricFields = {'eOspa', 'rmse', 'cardErr'};
localMetricNames = {'Local E-OSPA', 'Local RMSE', 'Local cardinality error'};

for levelIdx = 1:numel(levels)
    level = levels(levelIdx);
    fprintf('\n=====================================\n');
    fprintf('Communication level %d/%d: level=%d (%s)\n', ...
        levelIdx, numel(levels), level, levelLabels{levelIdx});
    fprintf('=====================================\n');

    [levelReportPath, levelSummary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
        numberOfTrials, baseSeed, useFixedSeed, commLevelOverrides(level), false, ...
        'fidFiaExistenceRefinement', struct(), armSelection);

    summary.levelSummaries{levelIdx} = levelSummary;
    summary.levelReportPaths{levelIdx} = levelReportPath;
    if levelIdx == 1
        summary.armNames = levelSummary.armNames;
        numMethods = numel(levelSummary.armNames);
        summary.trialSeeds = levelSummary.trialSeeds;
        summary.network = initMetricStats(networkMetricFields, numel(levels), numMethods);
        summary.local = initMetricStats(localMetricFields, numel(levels), numMethods);
        summary.runtime = initMetricStats({'filterSeconds'}, numel(levels), numMethods);
    end

    for metricIdx = 1:numel(networkMetricFields)
        fieldName = networkMetricFields{metricIdx};
        trialValues = levelSummary.consensusTrials.(fieldName);
        summary.network.(fieldName) = fillMetricStats(summary.network.(fieldName), ...
            levelIdx, trialValues);
    end

    localTrialValues = struct();
    localTrialValues.eOspa = computeTrialSensorMeans(levelSummary.localTrials.eOspa, false);
    localTrialValues.rmse = computeTrialSensorMeans(levelSummary.localTrials.rmse, true);
    localTrialValues.cardErr = computeTrialSensorMeans(levelSummary.localTrials.cardErr, false);
    for metricIdx = 1:numel(localMetricFields)
        fieldName = localMetricFields{metricIdx};
        summary.local.(fieldName) = fillMetricStats(summary.local.(fieldName), ...
            levelIdx, localTrialValues.(fieldName));
    end

    summary.runtime.filterSeconds = fillMetricStats(summary.runtime.filterSeconds, ...
        levelIdx, levelSummary.runtime.filterSeconds);
end

writeMetricCsv(outputCsvPath, summary, networkMetricFields, networkMetricNames, ...
    localMetricFields, localMetricNames);

reportPath = '';
if writeReport
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPath = fullfile(reportDir, sprintf( ...
        'GA_COMM_LEVEL_THREE_METHOD_N%d_SEED%d_%s.md', numberOfTrials, baseSeed, timestamp));
    writeAggregateReport(reportPath, summary, networkMetricFields, networkMetricNames, ...
        localMetricFields, localMetricNames);
    fprintf('Communication-level three-method report written: %s\n', reportPath);
end
fprintf('Communication-level three-method CSV written: %s\n', outputCsvPath);
end

function stats = initMetricStats(fieldNames, numLevels, numMethods)
stats = struct();
for fieldIdx = 1:numel(fieldNames)
    fieldName = fieldNames{fieldIdx};
    stats.(fieldName).mean = NaN(numLevels, numMethods);
    stats.(fieldName).std = NaN(numLevels, numMethods);
    stats.(fieldName).ciLow = NaN(numLevels, numMethods);
    stats.(fieldName).ciHigh = NaN(numLevels, numMethods);
    stats.(fieldName).n = zeros(numLevels, numMethods);
end
end

function stats = fillMetricStats(stats, levelIdx, trialValues)
for methodIdx = 1:size(trialValues, 2)
    values = trialValues(:, methodIdx);
    values = values(~isnan(values));
    n = numel(values);
    stats.n(levelIdx, methodIdx) = n;
    if n == 0
        continue;
    end
    metricMean = mean(values);
    if n > 1
        metricStd = std(values, 0);
        halfWidth = tCritical95(n - 1) * metricStd / sqrt(n);
    else
        metricStd = 0;
        halfWidth = 0;
    end
    stats.mean(levelIdx, methodIdx) = metricMean;
    stats.std(levelIdx, methodIdx) = metricStd;
    stats.ciLow(levelIdx, methodIdx) = metricMean - halfWidth;
    stats.ciHigh(levelIdx, methodIdx) = metricMean + halfWidth;
end
end

function values = computeTrialSensorMeans(metricValues, omitNan)
numTrials = size(metricValues, 1);
numArms = size(metricValues, 3);
values = NaN(numTrials, numArms);
for trial = 1:numTrials
    for armIdx = 1:numArms
        sensorValues = reshape(metricValues(trial, :, armIdx), 1, []);
        if omitNan
            sensorValues = sensorValues(~isnan(sensorValues));
        end
        if ~isempty(sensorValues)
            values(trial, armIdx) = mean(sensorValues);
        end
    end
end
end

function writeMetricCsv(csvPath, summary, networkMetricFields, networkMetricNames, ...
    localMetricFields, localMetricNames)
outDir = fileparts(csvPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
fid = fopen(csvPath, 'w');
if fid < 0
    error('Unable to open CSV for writing: %s', csvPath);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, 'metric_group,metric,level,level_label,method,mean,std,ci_low,ci_high,n\n');
for metricIdx = 1:numel(networkMetricFields)
    writeMetricRows(fid, 'network', networkMetricNames{metricIdx}, ...
        summary.network.(networkMetricFields{metricIdx}), summary);
end
for metricIdx = 1:numel(localMetricFields)
    writeMetricRows(fid, 'local', localMetricNames{metricIdx}, ...
        summary.local.(localMetricFields{metricIdx}), summary);
end
writeMetricRows(fid, 'runtime', 'Filter runtime', summary.runtime.filterSeconds, summary);
end

function writeMetricRows(fid, metricGroup, metricName, metricStats, summary)
for levelIdx = 1:numel(summary.levels)
    for methodIdx = 1:numel(summary.methodDisplayNames)
        fprintf(fid, '%s,%s,%d,%s,%s,%.9f,%.9f,%.9f,%.9f,%d\n', ...
            metricGroup, metricName, summary.levels(levelIdx), ...
            summary.levelLabels{levelIdx}, summary.methodDisplayNames{methodIdx}, ...
            metricStats.mean(levelIdx, methodIdx), metricStats.std(levelIdx, methodIdx), ...
            metricStats.ciLow(levelIdx, methodIdx), metricStats.ciHigh(levelIdx, methodIdx), ...
            metricStats.n(levelIdx, methodIdx));
    end
end
end

function writeAggregateReport(reportPath, summary, networkMetricFields, networkMetricNames, ...
    localMetricFields, localMetricNames)
fid = fopen(reportPath, 'w');
if fid < 0
    error('Unable to write report: %s', reportPath);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '# Communication-Level Three-Method Comparison\n\n');
fprintf(fid, '- Trials: %d\n', summary.numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', summary.baseSeed, summary.useFixedSeed);
fprintf(fid, '- trialSeeds: %s\n', mat2str(summary.trialSeeds));
fprintf(fid, '- CSV: `%s`\n\n', summary.outputCsvPath);

fprintf(fid, '## Network Metrics\n\n');
for metricIdx = 1:numel(networkMetricFields)
    writeMarkdownMetricTable(fid, networkMetricNames{metricIdx}, ...
        summary.network.(networkMetricFields{metricIdx}), summary);
end

fprintf(fid, '## Local Metrics\n\n');
for metricIdx = 1:numel(localMetricFields)
    writeMarkdownMetricTable(fid, localMetricNames{metricIdx}, ...
        summary.local.(localMetricFields{metricIdx}), summary);
end

fprintf(fid, '## Runtime\n\n');
writeMarkdownMetricTable(fid, 'Filter runtime', summary.runtime.filterSeconds, summary);
end

function writeMarkdownMetricTable(fid, metricName, metricStats, summary)
fprintf(fid, '### %s\n\n', metricName);
fprintf(fid, '| Level | Constraint | Method | Mean | Std | 95%% CI | N |\n');
fprintf(fid, '|--:|:--|:--|--:|--:|:--|--:|\n');
for levelIdx = 1:numel(summary.levels)
    for methodIdx = 1:numel(summary.methodDisplayNames)
        fprintf(fid, '| %d | %s | %s | %.6f | %.6f | [%.6f, %.6f] | %d |\n', ...
            summary.levels(levelIdx), summary.levelLabels{levelIdx}, ...
            summary.methodDisplayNames{methodIdx}, ...
            metricStats.mean(levelIdx, methodIdx), metricStats.std(levelIdx, methodIdx), ...
            metricStats.ciLow(levelIdx, methodIdx), metricStats.ciHigh(levelIdx, methodIdx), ...
            metricStats.n(levelIdx, methodIdx));
    end
end
fprintf(fid, '\n');
end

function commConfig = commLevelOverrides(level)
commConfig = struct();
commConfig.level = level;
commConfig.linkModel = 'fixed';
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.sensorWeights = ones(1, 8) / 8;
switch level
    case 0
        commConfig.globalMaxMeasurementsPerStep = inf;
        commConfig.pDrop = 0.0;
        commConfig.pDropBySensor = zeros(1, 8);
        commConfig.pDropLevels = [];
        commConfig.pDropLevelCounts = [];
        commConfig.maxOutageNodes = 0;
    case 1
        commConfig.globalMaxMeasurementsPerStep = 80;
        commConfig.pDrop = 0.0;
        commConfig.pDropBySensor = zeros(1, 8);
        commConfig.pDropLevels = [];
        commConfig.pDropLevelCounts = [];
        commConfig.maxOutageNodes = 0;
    case 2
        commConfig.globalMaxMeasurementsPerStep = 80;
        commConfig.pDrop = 0.2;
        commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
        commConfig.pDropLevelCounts = [1, 4, 1, 2];
        commConfig.maxOutageNodes = 0;
    case 3
        commConfig.globalMaxMeasurementsPerStep = 80;
        commConfig.pDrop = 0.2;
        commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
        commConfig.pDropLevelCounts = [1, 4, 1, 2];
        commConfig.maxOutageNodes = 1;
    otherwise
        error('Unsupported communication level: %d', level);
end
end

function tcrit = tCritical95(df)
if df <= 0
    tcrit = 0;
elseif df == 1
    tcrit = 12.706;
elseif df == 2
    tcrit = 4.303;
elseif df == 3
    tcrit = 3.182;
elseif df == 4
    tcrit = 2.776;
elseif df == 5
    tcrit = 2.571;
elseif df <= 10
    tableDf = [6 7 8 9 10];
    tableTc = [2.447 2.365 2.306 2.262 2.228];
    [~, idx] = min(abs(tableDf - df));
    tcrit = tableTc(idx);
elseif df <= 20
    tcrit = 2.086;
elseif df <= 30
    tcrit = 2.042;
elseif df <= 40
    tcrit = 2.021;
elseif df <= 60
    tcrit = 2.000;
else
    tcrit = 1.960;
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
