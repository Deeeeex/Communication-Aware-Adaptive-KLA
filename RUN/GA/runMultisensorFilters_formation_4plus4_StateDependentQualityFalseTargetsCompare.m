function [reportPath, summary] = runMultisensorFilters_formation_4plus4_StateDependentQualityFalseTargetsCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, adaptiveFusionOverrides, scenarioOverrides)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_STATEDEPENDENTQUALITYFALSETARGETSCOMPARE
% Compare fixed-weight GA against adaptive GA on the original 4+4 formation
% scene, with only two added conditions: state-dependent sensor quality and
% persistent false-target measurements.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 3;
end
if nargin < 2 || isempty(baseSeed)
    baseSeed = 9;
end
if nargin < 3 || isempty(useFixedSeed)
    useFixedSeed = true;
end
if nargin < 4 || isempty(writeReport)
    writeReport = true;
end
if nargin < 5 || isempty(adaptiveFusionOverrides)
    adaptiveFusionOverrides = struct();
end
if nargin < 6 || isempty(scenarioOverrides)
    scenarioOverrides = struct();
end

reportPath = '';
summary = struct();

simulationLength = getField(scenarioOverrides, 'simulationLength', 100);
numberOfSensors = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis';
leaderSensor = 8;

clutterRates = getField(scenarioOverrides, 'clutterRates', 3 * ones(1, numberOfSensors));
detectionProbabilities = getField(scenarioOverrides, 'detectionProbabilities', ...
    0.9 * ones(1, numberOfSensors));
q = getField(scenarioOverrides, 'q', 3 * ones(1, numberOfSensors));

adaptiveFusionConfig = buildAdaptiveFusionConfig();
adaptiveFusionConfig = mergeStructFields(adaptiveFusionConfig, adaptiveFusionOverrides);

commConfig = struct();
commConfig.level = 2;
commConfig.globalMaxMeasurementsPerStep = 80;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.maxOutageNodes = 1;
commConfig = mergeStructFields(commConfig, getField(scenarioOverrides, 'commConfig', struct()));

sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.0;
sensorMotionConfig.initialStates = buildSensorInitialStates();

targetFormationConfig = struct();
targetFormationConfig.targetFormationEnabled = true;
targetFormationConfig.targetFormationStaggeredBirths = true;
targetFormationConfig.targetFormationBirthInterval = getField(scenarioOverrides, 'targetBirthInterval', 8);
targetFormationConfig.targetFormationStartTime = 1;
targetFormationConfig.targetFormationLifeSpan = simulationLength;
targetFormationConfig.targetBirthStates = getField(scenarioOverrides, 'targetBirthStates', buildTargetBirthStates());
targetFormationConfig.targetFormationCount = size(targetFormationConfig.targetBirthStates, 2);

falseTargetConfig = buildFalseTargetConfig(simulationLength);
falseTargetConfig = mergeStructFields(falseTargetConfig, getField(scenarioOverrides, 'falseTargetConfig', struct()));

eOspaBase = zeros(numberOfTrials, numberOfSensors);
hOspaBase = zeros(numberOfTrials, numberOfSensors);
rmseBase = zeros(numberOfTrials, numberOfSensors);
cardErrBase = zeros(numberOfTrials, numberOfSensors);
eOspaAdaptive = zeros(numberOfTrials, numberOfSensors);
hOspaAdaptive = zeros(numberOfTrials, numberOfSensors);
rmseAdaptive = zeros(numberOfTrials, numberOfSensors);
cardErrAdaptive = zeros(numberOfTrials, numberOfSensors);
consOspaBase = zeros(numberOfTrials, 1);
consOspaAdaptive = zeros(numberOfTrials, 1);
consPosBase = zeros(numberOfTrials, 1);
consPosAdaptive = zeros(numberOfTrials, 1);
consCardBase = zeros(numberOfTrials, 1);
consCardAdaptive = zeros(numberOfTrials, 1);
falseMeasurementsBySensor = zeros(numberOfTrials, numberOfSensors);
falseMeasurementsTotal = zeros(numberOfTrials, 1);
baseMeasurementsTotal = zeros(numberOfTrials, 1);
allMeasurementsTotal = zeros(numberOfTrials, 1);

for trial = 1:numberOfTrials
    fprintf('State-dependent quality / false-target trial %d/%d\n', trial, numberOfTrials);
    if useFixedSeed
        rng(baseSeed + trial);
    end

    model = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'GA', 'LBP', 'Formation', ...
        sensorMotionConfig, targetFormationConfig);
    model.simulationLength = simulationLength;
    model.sensorCommRange = sensorCommRange;
    model.fusionWeighting = fusionWeighting;
    model.sensorFovEnabled = true;
    model.sensorFovHalfAngleDeg = 60;
    model.sensorFovRange = 60000;
    model.sensorQuality = mergeStructFields(buildSensorQualityConfig(), ...
        getField(scenarioOverrides, 'sensorQuality', struct()));

    [~, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
    model.sensorTrajectories = sensorTrajectories;
    baseMeasurementsTotal(trial) = countMeasurements(measurements);
    [measurements, falseStats] = injectFalseTargetMeasurements( ...
        measurements, model, sensorTrajectories, falseTargetConfig);
    falseMeasurementsBySensor(trial, :) = falseStats.bySensor;
    falseMeasurementsTotal(trial) = falseStats.total;
    allMeasurementsTotal(trial) = countMeasurements(measurements);

    [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);

    neighborMap = buildNeighborMap4Plus4(numberOfSensors);

    modelBase = model;
    modelBase.adaptiveFusion = adaptiveFusionConfig;
    modelBase.adaptiveFusion.enabled = false;
    [stateEstimatesBySensorBase, localModelsBase] = runDistributedLmbFilter( ...
        modelBase, measurementsDelivered, sensorTrajectories, neighborMap, commStats);

    modelAdaptive = model;
    modelAdaptive.adaptiveFusion = adaptiveFusionConfig;
    modelAdaptive.adaptiveFusion.enabled = true;
    [stateEstimatesBySensorAdaptive, localModelsAdaptive] = runDistributedLmbFilter( ...
        modelAdaptive, measurementsDelivered, sensorTrajectories, neighborMap, commStats);

    for s = 1:numberOfSensors
        [eBase, hBase, cardBase] = computeSimulationOspa(localModelsBase{s}, groundTruthRfs, stateEstimatesBySensorBase{s});
        [eAdaptive, hAdaptive, cardAdaptive] = computeSimulationOspa(localModelsAdaptive{s}, groundTruthRfs, stateEstimatesBySensorAdaptive{s});
        eOspaBase(trial, s) = mean(eBase);
        hOspaBase(trial, s) = mean(hBase);
        rmseBase(trial, s) = meanNoNan(computeSetRmseOverTime(stateEstimatesBySensorBase{s}, groundTruthRfs));
        cardErrBase(trial, s) = mean(abs(cardBase - groundTruthRfs.cardinality));
        eOspaAdaptive(trial, s) = mean(eAdaptive);
        hOspaAdaptive(trial, s) = mean(hAdaptive);
        rmseAdaptive(trial, s) = meanNoNan(computeSetRmseOverTime(stateEstimatesBySensorAdaptive{s}, groundTruthRfs));
        cardErrAdaptive(trial, s) = mean(abs(cardAdaptive - groundTruthRfs.cardinality));
    end

    [posBase, cardBase, ospaBase] = computeConsensusMetrics(stateEstimatesBySensorBase, modelBase);
    [posAdaptive, cardAdaptive, ospaAdaptive] = computeConsensusMetrics(stateEstimatesBySensorAdaptive, modelAdaptive);
    consOspaBase(trial) = mean(ospaBase);
    consOspaAdaptive(trial) = mean(ospaAdaptive);
    consPosBase(trial) = meanNoNan(posBase);
    consPosAdaptive(trial) = meanNoNan(posAdaptive);
    consCardBase(trial) = mean(cardBase);
    consCardAdaptive(trial) = mean(cardAdaptive);
end

fprintf('=====================================\n');
fprintf('GA state-dependent sensor quality + false targets (N=%d)\n', numberOfTrials);
fprintf('Control=fixed weights, Experiment=adaptive GA\n');
fprintf('False measurements per trial: %.2f\n', mean(falseMeasurementsTotal));
fprintf('=====================================\n');
fprintf('OSPA consensus error: %.6f -> %.6f\n', mean(consOspaBase), mean(consOspaAdaptive));
fprintf('Matched localization disagreement: %.6f -> %.6f\n', meanNoNan(consPosBase), meanNoNan(consPosAdaptive));
fprintf('Cardinality dispersion: %.6f -> %.6f\n', mean(consCardBase), mean(consCardAdaptive));

summary.scenario = struct();
summary.scenario.name = 'original 4+4 formation with state-dependent quality and persistent false targets';
summary.scenario.simulationLength = simulationLength;
summary.scenario.numberOfSensors = numberOfSensors;
summary.scenario.numberOfTrueTargets = targetFormationConfig.targetFormationCount;
summary.scenario.numberOfFalseTargets = size(falseTargetConfig.birthStates, 2);
summary.scenario.clutterRates = clutterRates;
summary.scenario.baseDetectionProbabilities = detectionProbabilities;
summary.scenario.baseMeasurementStd = q;
summary.scenario.sensorQuality = mergeStructFields(buildSensorQualityConfig(), ...
    getField(scenarioOverrides, 'sensorQuality', struct()));
summary.falseTargets.bySensor = falseMeasurementsBySensor;
summary.falseTargets.total = falseMeasurementsTotal;
summary.falseTargets.meanBySensor = mean(falseMeasurementsBySensor, 1);
summary.falseTargets.meanTotal = mean(falseMeasurementsTotal);
summary.measurements.baseTotal = baseMeasurementsTotal;
summary.measurements.falseTotal = falseMeasurementsTotal;
summary.measurements.allTotal = allMeasurementsTotal;
summary.measurements.falseFraction = falseMeasurementsTotal ./ max(allMeasurementsTotal, 1);
summary.measurements.meanBaseTotal = mean(baseMeasurementsTotal);
summary.measurements.meanAllTotal = mean(allMeasurementsTotal);
summary.measurements.meanFalseFraction = mean(summary.measurements.falseFraction);
summary.local.eOspaBase = mean(eOspaBase, 1);
summary.local.eOspaAdaptive = mean(eOspaAdaptive, 1);
summary.local.hOspaBase = mean(hOspaBase, 1);
summary.local.hOspaAdaptive = mean(hOspaAdaptive, 1);
summary.local.rmseBase = mean(rmseBase, 1);
summary.local.rmseAdaptive = mean(rmseAdaptive, 1);
summary.local.cardErrBase = mean(cardErrBase, 1);
summary.local.cardErrAdaptive = mean(cardErrAdaptive, 1);
summary.localTrials.eOspaBase = eOspaBase;
summary.localTrials.eOspaAdaptive = eOspaAdaptive;
summary.localTrials.hOspaBase = hOspaBase;
summary.localTrials.hOspaAdaptive = hOspaAdaptive;
summary.localTrials.rmseBase = rmseBase;
summary.localTrials.rmseAdaptive = rmseAdaptive;
summary.localTrials.cardErrBase = cardErrBase;
summary.localTrials.cardErrAdaptive = cardErrAdaptive;
summary.consensus.ospaBase = mean(consOspaBase);
summary.consensus.ospaAdaptive = mean(consOspaAdaptive);
summary.consensus.posBase = meanNoNan(consPosBase);
summary.consensus.posAdaptive = meanNoNan(consPosAdaptive);
summary.consensus.cardBase = mean(consCardBase);
summary.consensus.cardAdaptive = mean(consCardAdaptive);
summary.consensusTrials.ospaBase = consOspaBase;
summary.consensusTrials.ospaAdaptive = consOspaAdaptive;
summary.consensusTrials.posBase = consPosBase;
summary.consensusTrials.posAdaptive = consPosAdaptive;
summary.consensusTrials.cardBase = consCardBase;
summary.consensusTrials.cardAdaptive = consCardAdaptive;
summary.delta.ospa = summary.consensus.ospaAdaptive - summary.consensus.ospaBase;
summary.delta.pos = summary.consensus.posAdaptive - summary.consensus.posBase;
summary.delta.card = summary.consensus.cardAdaptive - summary.consensus.cardBase;
summary.delta.localEospa = mean(summary.local.eOspaAdaptive) - mean(summary.local.eOspaBase);
summary.delta.localHospa = mean(summary.local.hOspaAdaptive) - mean(summary.local.hOspaBase);
summary.delta.localRmse = mean(summary.local.rmseAdaptive) - mean(summary.local.rmseBase);
summary.delta.localCardErr = mean(summary.local.cardErrAdaptive) - mean(summary.local.cardErrBase);
summary.commConfig = commConfig;
summary.adaptiveFusionConfig = adaptiveFusionConfig;
summary.trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportName = sprintf('GA_STATE_DEP_QUALITY_FALSE_TARGETS_N%d_SEED%d_%s.md', ...
        numberOfTrials, baseSeed, timestamp);
    reportPath = fullfile(reportDir, reportName);
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
        sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
        commConfig, summary, eOspaBase, eOspaAdaptive, hOspaBase, hOspaAdaptive, ...
        rmseBase, rmseAdaptive, cardErrBase, cardErrAdaptive, consOspaBase, ...
        consOspaAdaptive, consPosBase, consPosAdaptive, consCardBase, consCardAdaptive);
    fprintf('Report written: %s\n', reportPath);
end
end

function cfg = buildAdaptiveFusionConfig()
cfg = struct( ...
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
    'spatialStructureStrength', 0.45, ...
    'existenceStructureStrength', 0.08, ...
    'structureReliabilityPower', 0.30, ...
    'structureReliabilityMinScore', 0.25, ...
    'useFreshness', false, ...
    'useHistory', false, ...
    'useNIS', true, ...
    'robustNIS', true, ...
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
end

function cfg = buildSensorQualityConfig()
cfg = struct();
cfg.enabled = true;
cfg.referenceRange = 180;
cfg.minDetectionProbability = 0.25;
cfg.detectionRangeDecay = 0.18;
cfg.detectionRangePower = 1.35;
cfg.edgeDetectionPenalty = 0.25;
cfg.anglePower = 2.0;
cfg.rangeNoiseScale = 0.35;
cfg.edgeNoiseScale = 0.8;
cfg.minCovarianceScale = 1.0;
cfg.maxCovarianceScale = 4.0;
end

function cfg = buildFalseTargetConfig(simulationLength)
cfg = struct();
cfg.birthStates = [ ...
     25,  95,  35;
     85,  15, -85;
     -0.10, -0.55, -0.10;
     -0.55, -0.05,  0.55];
cfg.startTimes = [15 30 45];
cfg.endTimes = [simulationLength, simulationLength, simulationLength];
cfg.detectionScale = 0.55;
cfg.maxDetectionProbability = 0.55;
end

function [measurements, stats] = injectFalseTargetMeasurements(measurements, model, sensorTrajectories, cfg)
stats = struct();
stats.bySensor = zeros(1, model.numberOfSensors);
stats.total = 0;
if isempty(cfg) || ~isfield(cfg, 'birthStates') || isempty(cfg.birthStates)
    return;
end

localModel = model;
localModel.sensorTrajectories = sensorTrajectories;
numFalseTargets = size(cfg.birthStates, 2);
simulationLength = size(measurements, 2);
for f = 1:numFalseTargets
    x = cfg.birthStates(:, f);
    startTime = max(1, round(cfg.startTimes(f)));
    endTime = min(simulationLength, round(cfg.endTimes(f)));
    for t = startTime:endTime
        if t > startTime
            x = model.A * x + model.u;
        end
        for s = 1:model.numberOfSensors
            [pdSensor, qSensor] = evaluateSensorQuality(localModel, s, x, t);
            falsePd = min(pdSensor * cfg.detectionScale, cfg.maxDetectionProbability);
            if rand < falsePd
                sensorPos = sensorTrajectories{s}(1:2, t);
                relativePos = x(1:2) - sensorPos;
                y = sensorPos + model.C{s} * [relativePos; 0; 0] + ...
                    chol(qSensor, 'lower') * randn(model.zDimension, 1);
                sensorMeasurements = measurements{s, t};
                sensorMeasurements{numel(sensorMeasurements) + 1} = y;
                measurements{s, t} = sensorMeasurements;
                stats.bySensor(s) = stats.bySensor(s) + 1;
                stats.total = stats.total + 1;
            end
        end
    end
end
end

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
    sensorCommRange, fusionWeighting, leaderSensor, adaptiveFusionConfig, ...
    commConfig, summary, eOspaBase, eOspaAdaptive, hOspaBase, hOspaAdaptive, ...
    rmseBase, rmseAdaptive, cardErrBase, cardErrAdaptive, consOspaBase, ...
    consOspaAdaptive, consPosBase, consPosAdaptive, consCardBase, consCardAdaptive)

fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf(fid, '# GA Original 4+4 + State-Dependent Sensor Quality + False Targets (%s)\n\n', timestamp);
fprintf(fid, 'Comparison order: fixed-weight GA -> adaptive GA\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
if useFixedSeed
    fprintf(fid, '- trialSeeds: %s\n', mat2str(computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)));
end
fprintf(fid, '- simulationLength: %d\n', summary.scenario.simulationLength);
fprintf(fid, '- trueTargets: %d\n', summary.scenario.numberOfTrueTargets);
fprintf(fid, '- falseTargets: %d\n', summary.scenario.numberOfFalseTargets);
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- comm level: %d\n', getField(commConfig, 'level', 0));
fprintf(fid, '- globalMaxMeasurementsPerStep: %.3f\n', getField(commConfig, 'globalMaxMeasurementsPerStep', inf));
fprintf(fid, '- linkModel: %s\n', getField(commConfig, 'linkModel', 'fixed'));
fprintf(fid, '- pDrop: %.3f\n', getField(commConfig, 'pDrop', 0));
fprintf(fid, '- clutterRates: %s\n', mat2str(summary.scenario.clutterRates, 3));
fprintf(fid, '- baseDetectionProbabilities: %s\n', mat2str(summary.scenario.baseDetectionProbabilities, 3));
fprintf(fid, '- baseMeasurementStd: %s\n\n', mat2str(summary.scenario.baseMeasurementStd, 3));

qcfg = summary.scenario.sensorQuality;
fprintf(fid, '## Sensor Quality Model\n');
fprintf(fid, '- referenceRange: %.3f\n', qcfg.referenceRange);
fprintf(fid, '- minDetectionProbability: %.3f\n', qcfg.minDetectionProbability);
fprintf(fid, '- detectionRangeDecay: %.3f\n', qcfg.detectionRangeDecay);
fprintf(fid, '- edgeDetectionPenalty: %.3f\n', qcfg.edgeDetectionPenalty);
fprintf(fid, '- rangeNoiseScale: %.3f\n', qcfg.rangeNoiseScale);
fprintf(fid, '- edgeNoiseScale: %.3f\n', qcfg.edgeNoiseScale);
fprintf(fid, '- maxCovarianceScale: %.3f\n\n', qcfg.maxCovarianceScale);

fprintf(fid, '## Adaptive Config\n');
fprintf(fid, '- useCovariance: %d\n', adaptiveFusionConfig.useCovariance);
fprintf(fid, '- useExistenceConfidence: %d\n', adaptiveFusionConfig.useExistenceConfidence);
fprintf(fid, '- useDecoupledKla: %d\n', adaptiveFusionConfig.useDecoupledKla);
fprintf(fid, '- useStructureAwareKla: %d\n', adaptiveFusionConfig.useStructureAwareKla);
fprintf(fid, '- useNIS: %d\n', adaptiveFusionConfig.useNIS);
fprintf(fid, '- robustNIS: %d\n\n', adaptiveFusionConfig.robustNIS);

fprintf(fid, '## False Measurements\n');
fprintf(fid, '- Mean total per trial: %.3f\n', summary.falseTargets.meanTotal);
fprintf(fid, '- Mean base measurements before false targets: %.3f\n', summary.measurements.meanBaseTotal);
fprintf(fid, '- Mean all measurements after false targets: %.3f\n', summary.measurements.meanAllTotal);
fprintf(fid, '- Mean false fraction: %.2f%%\n', 100 * summary.measurements.meanFalseFraction);
fprintf(fid, '- Mean by sensor: %s\n\n', mat2str(summary.falseTargets.meanBySensor, 4));

fprintf(fid, '## Per-Trial Network Disagreement Metrics\n');
fprintf(fid, '| Trial | Seed | Arm | OSPA | RMSE | Cardinality |\n');
fprintf(fid, '|------:|-----:|:----|-----:|-----:|------------:|\n');
trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
for trial = 1:numberOfTrials
    fprintf(fid, '| %d | %.0f | fixed-weight GA | %.6f | %.6f | %.6f |\n', ...
        trial, trialSeeds(trial), consOspaBase(trial), consPosBase(trial), consCardBase(trial));
    fprintf(fid, '| %d | %.0f | adaptive GA | %.6f | %.6f | %.6f |\n', ...
        trial, trialSeeds(trial), consOspaAdaptive(trial), consPosAdaptive(trial), consCardAdaptive(trial));
end
fprintf(fid, '\n');

fprintf(fid, '## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '- OSPA consensus error: %.3f -> %.3f\n', mean(consOspaBase), mean(consOspaAdaptive));
fprintf(fid, '- Matched localization disagreement: %.3f -> %.3f\n', meanNoNan(consPosBase), meanNoNan(consPosAdaptive));
fprintf(fid, '- Cardinality dispersion: %.3f -> %.3f\n\n', mean(consCardBase), mean(consCardAdaptive));

fprintf(fid, '## Paired Improvements Relative to Fixed-Weight GA\n');
writePairedImprovementLines(fid, 'OSPA consensus error', consOspaBase, consOspaAdaptive);
writePairedImprovementLines(fid, 'Matched localization disagreement', consPosBase, consPosAdaptive);
writePairedImprovementLines(fid, 'Cardinality dispersion', consCardBase, consCardAdaptive);
fprintf(fid, '\n');

fprintf(fid, '## Aggregated Local Metrics (mean across sensors and trials)\n');
fprintf(fid, '| Metric | Fixed-weight GA | Adaptive GA | Delta |\n');
fprintf(fid, '|:-------|----------------:|------------:|------:|\n');
writeLocalMetricRow(fid, 'E-OSPA', eOspaBase, eOspaAdaptive, false);
writeLocalMetricRow(fid, 'H-OSPA', hOspaBase, hOspaAdaptive, false);
writeLocalMetricRow(fid, 'RMSE', rmseBase, rmseAdaptive, true);
writeLocalMetricRow(fid, 'CardErr', cardErrBase, cardErrAdaptive, false);
fprintf(fid, '\n');

fprintf(fid, '## Local Tracking Metrics By Sensor (mean across trials)\n');
fprintf(fid, '| Sensor | E-OSPA (base) | E-OSPA (adaptive) | H-OSPA (base) | H-OSPA (adaptive) | RMSE (base) | RMSE (adaptive) | CardErr (base) | CardErr (adaptive) |\n');
fprintf(fid, '|------:|---------------:|------------------:|--------------:|-----------------:|------------:|---------------:|---------------:|------------------:|\n');
for s = 1:size(eOspaBase, 2)
    fprintf(fid, '| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n', ...
        s, mean(eOspaBase(:, s)), mean(eOspaAdaptive(:, s)), ...
        mean(hOspaBase(:, s)), mean(hOspaAdaptive(:, s)), ...
        meanNoNan(rmseBase(:, s)), meanNoNan(rmseAdaptive(:, s)), ...
        mean(cardErrBase(:, s)), mean(cardErrAdaptive(:, s)));
end

fclose(fid);
end

function writeLocalMetricRow(fid, metricName, baseValues, adaptiveValues, omitnanFlag)
baseMean = meanMatrix(baseValues, omitnanFlag);
adaptiveMean = meanMatrix(adaptiveValues, omitnanFlag);
fprintf(fid, '| %s | %.6f | %.6f | %.6f |\n', metricName, baseMean, adaptiveMean, adaptiveMean - baseMean);
end

function writePairedImprovementLines(fid, metricName, baseValues, adaptiveValues)
deltas = baseValues(:) - adaptiveValues(:);
deltas = deltas(isfinite(deltas));
if isempty(deltas)
    fprintf(fid, '- %s: insufficient finite samples\n', metricName);
    return;
end
baseMean = meanNoNan(baseValues);
reduction = mean(deltas);
if abs(baseMean) > eps
    pct = 100 * reduction / baseMean;
else
    pct = NaN;
end
fprintf(fid, '- %s: paired reduction %.6f (%.2f%%), wins %d/%d\n', ...
    metricName, reduction, pct, sum(deltas > 0), numel(deltas));
end

function count = countMeasurements(measurements)
count = 0;
for i = 1:numel(measurements)
    count = count + numel(measurements{i});
end
end

function seeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)
if useFixedSeed
    seeds = baseSeed + (1:numberOfTrials);
else
    seeds = NaN(1, numberOfTrials);
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
else
    dist = sqrt(mean(matched.^2));
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
        base = [0, -0.5, 0.5; 0.577, -0.289, -0.289];
    case 'leader3'
        base = [0, -0.75, -0.75, -1.5; 0, 0.6, -0.6, 0];
    otherwise
        base = zeros(2, count);
end
if size(base, 2) < count
    repeats = ceil(count / size(base, 2));
    base = repmat(base, 1, repeats);
end
offsets = spacing * base(:, 1:count);
end

function value = meanMatrix(values, omitnanFlag)
values = values(:);
if omitnanFlag
    values = values(isfinite(values));
end
if isempty(values)
    value = NaN;
else
    value = mean(values);
end
end

function value = meanNoNan(values)
value = meanMatrix(values, true);
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
