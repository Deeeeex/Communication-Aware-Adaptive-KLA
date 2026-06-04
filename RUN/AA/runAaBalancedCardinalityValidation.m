function [reportPath, summary] = runAaBalancedCardinalityValidation( ...
    numberOfTrials, baseSeed, useFixedSeed, aaControlOverrides, commConfigOverrides, writeReport, armSelection)
% RUNAABALANCEDCARDINALITYVALIDATION
% Formal AA diagnostic for the current Balanced and Cardinality-critical
% adaptive fusion modes.
%
% Default call:
%   [reportPath, summary] = runAaBalancedCardinalityValidation();
%
% Default design:
%   - 10 paired trials, deterministic seeds baseSeed + 1:baseSeed + N.
%   - Same 8-sensor formation and tiered packet-drop setting as the GA paper
%     ablation, but with lmbParallelUpdateMode fixed to AA.
%   - AA-specific safeguards are explicit in the report: stronger existence
%     pruning, lower Gaussian-mixture cap, and shorter diagnostic horizon.
%   - Arms: fixed AA, covariance-link AA, Balanced AA, Cardinality-critical AA.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 10;
end
if nargin < 2 || isempty(baseSeed)
    baseSeed = 1;
end
if nargin < 3 || isempty(useFixedSeed)
    useFixedSeed = true;
end
if nargin < 4 || isempty(aaControlOverrides)
    aaControlOverrides = struct();
end
if nargin < 5 || isempty(commConfigOverrides)
    commConfigOverrides = struct();
end
if nargin < 6 || isempty(writeReport)
    writeReport = true;
end
if nargin < 7
    armSelection = [];
end

numberOfTrials = max(0, round(numberOfTrials));
reportPath = '';
summary = struct();

aaControls = struct( ...
    'targetFormationLifeSpan', 24, ...
    'existenceThreshold', 0.03, ...
    'maximumNumberOfGmComponents', 3, ...
    'minimumTrajectoryLength', 10, ...
    'maximumNumberOfLbpIterations', 150, ...
    'lbpConvergenceTolerance', 1e-4, ...
    'progressEverySteps', 10, ...
    'saveMat', true, ...
    'saveCheckpoints', true);
aaControls = mergeStructFields(aaControls, aaControlOverrides);
aaControls.targetFormationLifeSpan = max(1, round(aaControls.targetFormationLifeSpan));
aaControls.maximumNumberOfGmComponents = max(1, round(aaControls.maximumNumberOfGmComponents));
aaControls.minimumTrajectoryLength = max(1, round(aaControls.minimumTrajectoryLength));
aaControls.maximumNumberOfLbpIterations = max(1, round(aaControls.maximumNumberOfLbpIterations));
aaControls.progressEverySteps = max(0, round(aaControls.progressEverySteps));

leaderSensor = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis';
numberOfSensors = 8;
clutterRates = 3 * ones(1, numberOfSensors);
detectionProbabilities = 0.9 * ones(1, numberOfSensors);
q = 3 * ones(1, numberOfSensors);

baseAdaptiveFusionConfig = buildBaseAdaptiveFusionConfig(aaControls);
arms = buildAaArms(baseAdaptiveFusionConfig);
arms = selectArms(arms, armSelection);
armNames = {arms.name};
numArms = numel(arms);

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
commConfig = mergeStructFields(commConfig, commConfigOverrides);

sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.0;
sensorMotionConfig.initialStates = buildSensorInitialStates();

targetFormationConfig = struct();
targetFormationConfig.targetFormationEnabled = true;
targetFormationConfig.targetFormationStaggeredBirths = true;
targetFormationConfig.targetFormationBirthInterval = 8;
targetFormationConfig.targetFormationStartTime = 1;
targetFormationConfig.targetFormationLifeSpan = aaControls.targetFormationLifeSpan;
targetFormationConfig.targetBirthStates = buildTargetBirthStates();
targetFormationConfig.targetFormationCount = size(targetFormationConfig.targetBirthStates, 2);

eOspa = zeros(numberOfTrials, numberOfSensors, numArms);
hOspa = zeros(numberOfTrials, numberOfSensors, numArms);
rmse = zeros(numberOfTrials, numberOfSensors, numArms);
cardErr = zeros(numberOfTrials, numberOfSensors, numArms);
consOspa = zeros(numberOfTrials, numArms);
consPos = zeros(numberOfTrials, numArms);
consCard = zeros(numberOfTrials, numArms);
filterRuntimeSeconds = zeros(numberOfTrials, numArms);
pDropBySensorTrials = zeros(numberOfTrials, numberOfSensors);
samplingStatsLast = struct();

reportDir = fullfile(projectRoot, 'RUN', 'AA');
if ~exist(reportDir, 'dir')
    mkdir(reportDir);
end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
reportPath = fullfile(reportDir, sprintf('AA_BALANCED_CARDINALITY_VALIDATION_N%d_SEED%d_%s.md', ...
    numberOfTrials, baseSeed, timestamp));
matPath = fullfile(reportDir, sprintf('AA_BALANCED_CARDINALITY_VALIDATION_N%d_SEED%d_%s.mat', ...
    numberOfTrials, baseSeed, timestamp));
checkpointPath = fullfile(reportDir, sprintf('AA_BALANCED_CARDINALITY_VALIDATION_N%d_SEED%d_%s_checkpoint.mat', ...
    numberOfTrials, baseSeed, timestamp));

for trial = 1:numberOfTrials
    fprintf('AA validation trial %d/%d\n', trial, numberOfTrials);
    if useFixedSeed
        rng(baseSeed + trial);
    end

    model = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'AA', 'LBP', 'Formation', ...
        sensorMotionConfig, targetFormationConfig);
    model.sensorCommRange = sensorCommRange;
    model.fusionWeighting = fusionWeighting;
    model.sensorFovEnabled = true;
    model.sensorFovHalfAngleDeg = 60;
    model.sensorFovRange = 60000;
    model = applyAaControlsToModel(model, aaControls);

    [~, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
    model.sensorTrajectories = sensorTrajectories;

    [measurementsForComm, samplingStats] = applyOptionalMultiRateSchedule(measurements, commConfig);
    [measurementsDelivered, commStats] = applyCommunicationModel(measurementsForComm, model, commConfig);
    commStats = attachSamplingStats(commStats, samplingStats);
    samplingStatsLast = samplingStats;
    pDropBySensorTrials(trial, :) = reshape(commStats.pDropBySensor, 1, []);

    for armIdx = 1:numArms
        fprintf('  Arm %d/%d: %s\n', armIdx, numArms, arms(armIdx).name);
        armModel = model;
        armModel.adaptiveFusion = arms(armIdx).adaptiveFusion;
        neighborMap = buildNeighborMap4Plus4(numberOfSensors);
        runtimeStart = tic();
        [stateEstimatesBySensor, localModels] = runDistributedLmbFilter( ...
            armModel, measurementsDelivered, sensorTrajectories, neighborMap, commStats);
        filterRuntimeSeconds(trial, armIdx) = toc(runtimeStart);
        fprintf('    Filter runtime: %.3f s\n', filterRuntimeSeconds(trial, armIdx));

        for s = 1:numberOfSensors
            [eArm, hArm, cardArm] = computeSimulationOspa(localModels{s}, groundTruthRfs, stateEstimatesBySensor{s});
            eOspa(trial, s, armIdx) = mean(eArm);
            hOspa(trial, s, armIdx) = mean(hArm);
            rmse(trial, s, armIdx) = mean(computeSetRmseOverTime(stateEstimatesBySensor{s}, groundTruthRfs), 'omitnan');
            cardErr(trial, s, armIdx) = mean(abs(reshape(cardArm, 1, []) - reshape(groundTruthRfs.cardinality, 1, [])));
        end

        [posArm, cardConsensusArm, ospaArm] = computeConsensusMetrics(stateEstimatesBySensor, armModel);
        consOspa(trial, armIdx) = mean(ospaArm);
        consPos(trial, armIdx) = mean(posArm, 'omitnan');
        consCard(trial, armIdx) = mean(cardConsensusArm);

        if aaControls.saveCheckpoints
            save(checkpointPath, 'trial', 'armIdx', 'armNames', 'aaControls', 'commConfig', ...
                'pDropBySensorTrials', 'consOspa', 'consPos', 'consCard', ...
                'eOspa', 'hOspa', 'rmse', 'cardErr', 'filterRuntimeSeconds', '-v7');
        end
    end
end

fprintf('=====================================\n');
fprintf('AA Balanced/Cardinality Validation (N=%d)\n', numberOfTrials);
fprintf('Order: %s\n', strjoin(armNames, ' -> '));
fprintf('AA controls: existenceThreshold=%.4f, maxGM=%d, minTrajectory=%d, lifeSpan=%d\n', ...
    aaControls.existenceThreshold, aaControls.maximumNumberOfGmComponents, ...
    aaControls.minimumTrajectoryLength, aaControls.targetFormationLifeSpan);
fprintf('=====================================\n');
for armIdx = 1:numArms
    fprintf('%s: OSPA %.6f, Loc %.6f, Card %.6f, Runtime %.3f s\n', arms(armIdx).name, ...
        mean(consOspa(:, armIdx)), mean(consPos(:, armIdx), 'omitnan'), ...
        mean(consCard(:, armIdx)), mean(filterRuntimeSeconds(:, armIdx)));
end

summary.armNames = armNames;
summary.consensus.ospa = mean(consOspa, 1);
summary.consensus.pos = mean(consPos, 1, 'omitnan');
summary.consensus.card = mean(consCard, 1);
summary.consensusTrials.ospa = consOspa;
summary.consensusTrials.pos = consPos;
summary.consensusTrials.card = consCard;
summary.local.meanAcrossSensors.eOspa = computePerArmGlobalMeans(eOspa, false);
summary.local.meanAcrossSensors.hOspa = computePerArmGlobalMeans(hOspa, false);
summary.local.meanAcrossSensors.rmse = computePerArmGlobalMeans(rmse, true);
summary.local.meanAcrossSensors.cardErr = computePerArmGlobalMeans(cardErr, false);
summary.localTrials.eOspa = eOspa;
summary.localTrials.hOspa = hOspa;
summary.localTrials.rmse = rmse;
summary.localTrials.cardErr = cardErr;
summary.runtime.filterSeconds = filterRuntimeSeconds;
summary.runtime.meanFilterSeconds = mean(filterRuntimeSeconds, 1);
summary.runtime.stdFilterSeconds = std(filterRuntimeSeconds, 0, 1);
summary.runtime.meanSecondsPerStep = summary.runtime.meanFilterSeconds ./ aaControls.targetFormationLifeSpan;
summary.runtime.relativeToFixed = summary.runtime.meanFilterSeconds ./ max(summary.runtime.meanFilterSeconds(1), eps);
summary.trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
summary.pDropBySensorTrials = pDropBySensorTrials;
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);
summary.commConfig = commConfig;
summary.aaControls = aaControls;
summary.samplingStats = samplingStatsLast;
summary.arms = arms;
summary.reportPath = reportPath;
summary.matPath = matPath;
summary.checkpointPath = checkpointPath;

if writeReport
    writeAaValidationReport(reportPath, summary, numberOfTrials, baseSeed, useFixedSeed, ...
        sensorCommRange, fusionWeighting, leaderSensor, arms, consOspa, consPos, consCard, ...
        eOspa, hOspa, rmse, cardErr, filterRuntimeSeconds);
    fprintf('Report written: %s\n', reportPath);
end

if aaControls.saveMat
    save(matPath, 'summary', '-v7');
    fprintf('MAT summary written: %s\n', matPath);
end
end

function cfg = buildBaseAdaptiveFusionConfig(aaControls)
cfg = struct( ...
    'enabled', true, ...
    'method', 'factorized', ...
    'emaAlpha', 0.7, ...
    'minWeight', 0.05, ...
    'useCovariance', true, ...
    'useLinkQuality', true, ...
    'useExistenceConfidence', false, ...
    'existenceConfidenceMinScore', 0.6, ...
    'existenceConfidencePower', 1.0, ...
    'useDecoupledKla', false, ...
    'spatialEmaAlpha', 0.7, ...
    'existenceEmaAlpha', 0.7, ...
    'spatialMinWeight', 0.05, ...
    'existenceMinWeight', 0.05, ...
    'spatialCovariancePower', 1.0, ...
    'spatialLinkQualityPower', 1.0, ...
    'existenceLinkQualityPower', 1.0, ...
    'existenceConfidenceWeightPower', 1.0, ...
    'spatialDecouplingStrength', 1.0, ...
    'existenceDecouplingStrength', 1.0, ...
    'useStructureAwareKla', false, ...
    'usePosteriorStructureConsistency', true, ...
    'spatialStructureStrength', 0.0, ...
    'existenceStructureStrength', 0.0, ...
    'structureReliabilityPower', 0.0, ...
    'structureReliabilityMinScore', 0.25, ...
    'useFidFiaExistence', false, ...
    'fidFiaExistenceStrength', 0.5, ...
    'fidFiaExistencePower', 1.0, ...
    'fidFiaExistenceMinScore', 0.4, ...
    'fidFiaQuadraturePoints', 3, ...
    'fidFiaUseDetectionProbability', true, ...
    'fidFiaUseExistenceWeight', true, ...
    'fidFiaUseEma', false, ...
    'fidFiaMinWeight', 0.0, ...
    'useFreshness', false, ...
    'useHistory', false, ...
    'useNIS', false, ...
    'progressEverySteps', aaControls.progressEverySteps);
end

function arms = buildAaArms(baseCfg)
arms = repmat(struct('name', '', 'adaptiveFusion', struct()), 1, 4);

cfg = baseCfg;
cfg.enabled = false;
cfg.useCovariance = false;
cfg.useLinkQuality = false;
cfg.useExistenceConfidence = false;
cfg.useDecoupledKla = false;
cfg.useStructureAwareKla = false;
cfg.useFidFiaExistence = false;
arms(1).name = 'fixed AA';
arms(1).adaptiveFusion = cfg;

cfg = baseCfg;
cfg.enabled = true;
cfg.method = 'factorized';
cfg.useCovariance = true;
cfg.useLinkQuality = true;
cfg.useExistenceConfidence = false;
cfg.useDecoupledKla = false;
cfg.useStructureAwareKla = false;
cfg.useFidFiaExistence = false;
arms(2).name = 'covariance-link AA';
arms(2).adaptiveFusion = cfg;

cfg = baseCfg;
cfg.enabled = true;
cfg.method = 'factorized';
cfg.useCovariance = true;
cfg.useLinkQuality = true;
cfg.useExistenceConfidence = true;
cfg.useDecoupledKla = true;
cfg.useStructureAwareKla = true;
cfg.usePosteriorStructureConsistency = false;
cfg.useFidFiaExistence = false;
cfg.existenceConfidenceMinScore = 0.85;
cfg.existenceConfidencePower = 2.0;
cfg.spatialDecouplingStrength = 0.5;
cfg.existenceDecouplingStrength = 0.15;
cfg.spatialStructureStrength = 0.45;
cfg.existenceStructureStrength = 0.08;
cfg.structureReliabilityPower = 0.30;
arms(3).name = 'Balanced AA';
arms(3).adaptiveFusion = cfg;

% AA consumes existence weights through a linear Bernoulli average.  The GA
% paper setting uses hard existence-branch suppression for Cardinality-critical
% mode, but that is too brittle for AA because the spatial branch still keeps
% mixture components from all active neighbours.  Use FID-FIA as a bounded
% cardinality cue here, while retaining the same active-neighbour floor and EMA
% stability as Balanced AA.
cfg.useFidFiaExistence = true;
cfg.fidFiaExistenceStrength = 1.0;
cfg.fidFiaExistenceMinScore = 0.4;
cfg.existenceEmaAlpha = 0.7;
cfg.existenceMinWeight = 0.05;
arms(4).name = 'Cardinality-critical AA';
arms(4).adaptiveFusion = cfg;
end

function model = applyAaControlsToModel(model, aaControls)
model.existenceThreshold = aaControls.existenceThreshold;
model.maximumNumberOfGmComponents = aaControls.maximumNumberOfGmComponents;
model.minimumTrajectoryLength = aaControls.minimumTrajectoryLength;
model.maximumNumberOfLbpIterations = aaControls.maximumNumberOfLbpIterations;
model.lbpConvergenceTolerance = aaControls.lbpConvergenceTolerance;
end

function arms = selectArms(arms, armSelection)
if nargin < 2 || isempty(armSelection)
    return;
end
if isnumeric(armSelection)
    armIdx = unique(max(1, min(numel(arms), round(armSelection(:)'))));
    arms = arms(armIdx);
    return;
end
if ischar(armSelection)
    requested = {armSelection};
elseif iscell(armSelection)
    requested = armSelection;
else
    requested = cellstr(armSelection);
end
matched = false(1, numel(arms));
for i = 1:numel(requested)
    query = lower(strtrim(char(requested{i})));
    for armIdx = 1:numel(arms)
        name = lower(arms(armIdx).name);
        if strcmp(query, 'final') && armIdx == numel(arms)
            matched(armIdx) = true;
        elseif strcmp(query, 'new') && armIdx >= 3
            matched(armIdx) = true;
        elseif ~isempty(strfind(name, query))
            matched(armIdx) = true;
        end
    end
end
if any(matched)
    arms = arms(matched);
end
end

function writeAaValidationReport(reportPath, summary, numberOfTrials, baseSeed, useFixedSeed, ...
    sensorCommRange, fusionWeighting, leaderSensor, arms, consOspa, consPos, consCard, ...
    eOspa, hOspa, rmse, cardErr, filterRuntimeSeconds)
fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end

aaControls = summary.aaControls;
commConfig = summary.commConfig;
armNames = {arms.name};
baselineName = arms(1).name;

fprintf(fid, '# AA Balanced/Cardinality Validation\n\n');
fprintf(fid, 'Generated at: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Comparison order: %s\n\n', strjoin(armNames, ' -> '));
fprintf(fid, '## Scope\n');
fprintf(fid, 'This is an AA-specific diagnostic, not a replacement for the GA/KLA paper main table. It tests whether the current Balanced and Cardinality-critical adaptive weights remain useful when consumed by AA-LMB fusion.\n\n');

fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
if useFixedSeed
    fprintf(fid, '- trialSeeds: %s\n', mat2str(computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)));
end
fprintf(fid, '- lmbParallelUpdateMode: AA\n');
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- targetFormationLifeSpan: %d\n', aaControls.targetFormationLifeSpan);
fprintf(fid, '- existenceThreshold: %.6f\n', aaControls.existenceThreshold);
fprintf(fid, '- maximumNumberOfGmComponents: %d\n', aaControls.maximumNumberOfGmComponents);
fprintf(fid, '- minimumTrajectoryLength: %d\n', aaControls.minimumTrajectoryLength);
fprintf(fid, '- maximumNumberOfLbpIterations: %d\n', aaControls.maximumNumberOfLbpIterations);
fprintf(fid, '- lbpConvergenceTolerance: %.3g\n', aaControls.lbpConvergenceTolerance);
fprintf(fid, '- linkModel: %s\n', getField(commConfig, 'linkModel', 'fixed'));
fprintf(fid, '- pDropLevels: %s\n', mat2str(getField(commConfig, 'pDropLevels', []), 3));
fprintf(fid, '- pDropLevelCounts: %s\n\n', mat2str(getField(commConfig, 'pDropLevelCounts', [])));

fprintf(fid, '## Arm Configs\n');
for armIdx = 1:numel(arms)
    cfg = arms(armIdx).adaptiveFusion;
    fprintf(fid, '### %s\n', arms(armIdx).name);
    fprintf(fid, '- enabled: %d\n', getField(cfg, 'enabled', false));
    fprintf(fid, '- method: %s\n', char(getField(cfg, 'method', 'factorized')));
    fprintf(fid, '- useCovariance: %d\n', getField(cfg, 'useCovariance', false));
    fprintf(fid, '- useLinkQuality: %d\n', getField(cfg, 'useLinkQuality', false));
    fprintf(fid, '- useExistenceConfidence: %d\n', getField(cfg, 'useExistenceConfidence', false));
    fprintf(fid, '- useDecoupledKla: %d\n', getField(cfg, 'useDecoupledKla', false));
    fprintf(fid, '- useStructureAwareKla: %d\n', getField(cfg, 'useStructureAwareKla', false));
    fprintf(fid, '- useFidFiaExistence: %d\n', getField(cfg, 'useFidFiaExistence', false));
    fprintf(fid, '- existenceConfidenceMinScore: %.3f\n', getField(cfg, 'existenceConfidenceMinScore', 0));
    fprintf(fid, '- existenceConfidencePower: %.3f\n', getField(cfg, 'existenceConfidencePower', 0));
    fprintf(fid, '- spatialDecouplingStrength: %.3f\n', getField(cfg, 'spatialDecouplingStrength', 0));
    fprintf(fid, '- existenceDecouplingStrength: %.3f\n', getField(cfg, 'existenceDecouplingStrength', 0));
    fprintf(fid, '- spatialStructureStrength: %.3f\n', getField(cfg, 'spatialStructureStrength', 0));
    fprintf(fid, '- existenceStructureStrength: %.3f\n', getField(cfg, 'existenceStructureStrength', 0));
    fprintf(fid, '- fidFiaExistenceStrength: %.3f\n', getField(cfg, 'fidFiaExistenceStrength', 0));
    fprintf(fid, '- fidFiaExistenceMinScore: %.3f\n', getField(cfg, 'fidFiaExistenceMinScore', 0));
    fprintf(fid, '- existenceMinWeight: %.3f\n\n', getField(cfg, 'existenceMinWeight', 0));
end

fprintf(fid, '## Per-Trial pDropBySensor\n');
for trial = 1:size(summary.pDropBySensorTrials, 1)
    fprintf(fid, '- Trial %d: %s\n', trial, mat2str(summary.pDropBySensorTrials(trial, :), 4));
end
fprintf(fid, '\n');

fprintf(fid, '## Per-Trial Network Disagreement Metrics\n');
fprintf(fid, '| Trial | Seed | Arm | OSPA | Loc. disag. | Card. disp. |\n');
fprintf(fid, '|------:|-----:|:----|-----:|------------:|------------:|\n');
trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
for trial = 1:numberOfTrials
    for armIdx = 1:numel(arms)
        fprintf(fid, '| %d | %.0f | %s | %.6f | %.6f | %.6f |\n', ...
            trial, trialSeeds(trial), arms(armIdx).name, ...
            consOspa(trial, armIdx), consPos(trial, armIdx), consCard(trial, armIdx));
    end
end
fprintf(fid, '\n');

fprintf(fid, '## Network Disagreement Metrics\n');
fprintf(fid, '| Arm | OSPA | Loc. disag. | Card. disp. |\n');
fprintf(fid, '|:----|-----:|------------:|------------:|\n');
for armIdx = 1:numel(arms)
    fprintf(fid, '| %s | %.6f | %.6f | %.6f |\n', arms(armIdx).name, ...
        mean(consOspa(:, armIdx)), mean(consPos(:, armIdx), 'omitnan'), mean(consCard(:, armIdx)));
end
fprintf(fid, '\n');
writeMetricStatsTable(fid, armNames, {'OSPA', 'Loc. disag.', 'Card. disp.'}, ...
    {consOspa, consPos, consCard}, [false, true, false]);

if numel(arms) >= 2
    fprintf(fid, '\n## Paired Improvements Relative to %s\n', baselineName);
    writePairedImprovementTable(fid, armNames, {'OSPA', 'Loc. disag.', 'Card. disp.'}, ...
        {consOspa, consPos, consCard}, [false, true, false]);
end

fprintf(fid, '\n## Local Tracking Metrics\n');
fprintf(fid, '| Arm | E-OSPA | RMSE | CardErr |\n');
fprintf(fid, '|:----|-------:|-----:|--------:|\n');
for armIdx = 1:numel(arms)
    fprintf(fid, '| %s | %.6f | %.6f | %.6f |\n', arms(armIdx).name, ...
        computeGlobalMean(eOspa(:, :, armIdx), false), ...
        computeGlobalMean(rmse(:, :, armIdx), true), ...
        computeGlobalMean(cardErr(:, :, armIdx), false));
end
fprintf(fid, '\n');
localEOspaTrial = computeTrialSensorMeans(eOspa, false);
localRmseTrial = computeTrialSensorMeans(rmse, true);
localCardTrial = computeTrialSensorMeans(cardErr, false);
writeMetricStatsTable(fid, armNames, {'E-OSPA', 'RMSE', 'CardErr'}, ...
    {localEOspaTrial, localRmseTrial, localCardTrial}, [false, true, false]);

if numel(arms) >= 2
    fprintf(fid, '\n## Paired Local-Metric Improvements Relative to %s\n', baselineName);
    writePairedImprovementTable(fid, armNames, {'E-OSPA', 'RMSE', 'CardErr'}, ...
        {localEOspaTrial, localRmseTrial, localCardTrial}, [false, true, false]);
end

fprintf(fid, '\n## Runtime\n');
writeRuntimeCostSummaryTable(fid, armNames, filterRuntimeSeconds, aaControls.targetFormationLifeSpan, baselineName);

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

function writePairedImprovementTable(fid, armNames, metricNames, metricArrays, omitnanFlags)
fprintf(fid, '| Arm | Metric | Paired reduction | 95%% CI | Reduction | Wins | Sign-test p |\n');
fprintf(fid, '|:----|:-------|-----------------:|:-------|----------:|-----:|------------:|\n');
for metricIdx = 1:numel(metricNames)
    data = metricArrays{metricIdx};
    baseline = data(:, 1);
    for armIdx = 2:numel(armNames)
        candidate = data(:, armIdx);
        valid = isfinite(baseline) & isfinite(candidate);
        if ~omitnanFlags(metricIdx)
            valid = valid & ~isnan(baseline) & ~isnan(candidate);
        end
        deltas = baseline(valid) - candidate(valid);
        stats = summarizeVector(deltas, true);
        baseStats = summarizeVector(baseline(valid), true);
        if isfinite(baseStats.mean) && abs(baseStats.mean) > eps
            pct = 100 * stats.mean / baseStats.mean;
        else
            pct = NaN;
        end
        wins = sum(deltas > 0);
        pValue = signTestPvalue(deltas);
        fprintf(fid, '| %s | %s | %.6f +/- %.6f | [%.6f, %.6f] | %.2f%% | %d/%d | %.4g |\n', ...
            armNames{armIdx}, metricNames{metricIdx}, stats.mean, stats.std, ...
            stats.ciLow, stats.ciHigh, pct, wins, stats.n, pValue);
    end
end
end

function writeRuntimeCostSummaryTable(fid, armNames, runtimeSeconds, simulationLength, baselineName)
fixedRuntime = runtimeSeconds(:, 1);
fprintf(fid, '| Arm | Filter runtime (s) | Runtime/step (s) | Relative to %s | N |\n', baselineName);
fprintf(fid, '|:----|-------------------:|-----------------:|------------------:|--:|\n');
for armIdx = 1:numel(armNames)
    values = runtimeSeconds(:, armIdx);
    stats = summarizeVector(values, true);
    if armIdx == 1
        ratioMean = 1.0;
    else
        validRatio = isfinite(values) & isfinite(fixedRuntime) & fixedRuntime > eps;
        if any(validRatio)
            ratioStats = summarizeVector(values(validRatio) ./ fixedRuntime(validRatio), true);
            ratioMean = ratioStats.mean;
        else
            ratioMean = NaN;
        end
    end
    fprintf(fid, '| %s | %.6f +/- %.6f | %.6f | %.3fx | %d |\n', ...
        armNames{armIdx}, stats.mean, stats.std, stats.mean / simulationLength, ratioMean, stats.n);
end
end

function trialMeans = computeTrialSensorMeans(values, omitnanFlag)
numTrials = size(values, 1);
numArms = size(values, 3);
trialMeans = zeros(numTrials, numArms);
for trial = 1:numTrials
    for armIdx = 1:numArms
        sensorValues = reshape(values(trial, :, armIdx), [], 1);
        stats = summarizeVector(sensorValues, omitnanFlag);
        trialMeans(trial, armIdx) = stats.mean;
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
table = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, ...
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, ...
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];
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

function values = computePerArmGlobalMeans(arrayLike, omitnanFlag)
numArms = size(arrayLike, 3);
values = zeros(1, numArms);
for armIdx = 1:numArms
    values(armIdx) = computeGlobalMean(arrayLike(:, :, armIdx), omitnanFlag);
end
end

function value = computeGlobalMean(arrayLike, omitnanFlag)
values = arrayLike(:);
if nargin >= 2 && omitnanFlag
    value = mean(values, 'omitnan');
else
    value = mean(values);
end
end

function [measurementsForComm, samplingStats] = applyOptionalMultiRateSchedule(measurements, commConfig)
measurementsForComm = measurements;
samplingStats = struct();
if ~getField(commConfig, 'enableMultiRate', false)
    return;
end
samplingPeriods = getField(commConfig, 'samplingPeriods', []);
if isempty(samplingPeriods)
    return;
end
phaseOffsets = getField(commConfig, 'samplingPhaseOffsets', []);
[measurementsForComm, samplingStats] = applyMultiRateSensorSchedule( ...
    measurements, samplingPeriods, phaseOffsets);
end

function commStats = attachSamplingStats(commStats, samplingStats)
if nargin < 2 || ~isstruct(samplingStats) || isempty(fieldnames(samplingStats))
    return;
end
commStats.sensorSampleMask = samplingStats.sensorSampleMask;
commStats.sensorSampleAge = samplingStats.sensorSampleAge;
commStats.droppedBySchedule = samplingStats.droppedBySchedule;
commStats.samplingPeriods = samplingStats.samplingPeriods;
commStats.samplingPhaseOffsets = samplingStats.samplingPhaseOffsets;
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
switch lower(formationType)
    case 'triangle'
        base = [0, -0.5, 0.5; 0, -0.866, -0.866];
    case 'leader3'
        base = [0, -1, -1, -2; 0, -0.7, 0.7, 0];
    otherwise
        base = [0, -1, 1; 0, -1, -1];
end
if size(base, 2) < count
    base = [base, zeros(2, count - size(base, 2))];
end
offsets = spacing * base(:, 1:count);
end

function value = getField(s, fieldName, defaultValue)
if nargin < 3
    defaultValue = [];
end
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end
