function [reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, commConfigOverrides, writeReport, finalArmMode, adaptiveFusionOverrides, armSelection)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_TIEREDLINKABLATION
% Ablation under tiered packet-drop configuration.
%
% 文件导读：
%   这是当前 GA-LMB 动态权重主实验/消融入口。它本身不实现滤波器细节，
%   而是负责把“场景、通信、实验 arm、分布式邻域、指标和报告”串起来。
%
% 主调用链：
%   1. 本脚本生成 8-sensor / 10-target formation 场景配置；
%   2. generateMultisensorModel 生成滤波模型；
%   3. generateMultisensorGroundTruth 生成 truth、measurements 和 sensor trajectories；
%   4. applyCommunicationModel 注入 tiered packet-drop，得到 delivered measurements；
%   5. buildArms 根据 finalArmMode 构造 Fixed / FID-FIA / Balanced / Cardinality-critical 等 arm；
%   6. 每个 arm 调 runDistributedLmbFilter；
%   7. runDistributedLmbFilter 为每个 sensor 构造 local neighbor model；
%   8. runParallelUpdateLmbFilter 在每个时刻调用 computeAdaptiveFusionWeights；
%   9. gaLmbTrackMerging 消费 spatial/existence weights 完成 posterior fusion；
%  10. 本脚本再计算 local tracking metrics、network disagreement metrics、runtime 和 markdown report。
%
% 阅读建议：
%   先看前 300 行主函数，理解“实验如何被组织”；再看 buildArms，理解
%   每个对照 arm 到底打开哪些 adaptiveFusion 开关；最后看 writeAblationReport
%   和 computeConsensusMetrics，理解结果表里的指标从哪里来。

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 1;
end
if nargin < 2 || isempty(baseSeed)
    baseSeed = 1;
end
if nargin < 3 || isempty(useFixedSeed)
    useFixedSeed = true;
end
if nargin < 4 || isempty(commConfigOverrides)
    commConfigOverrides = struct();
end
if nargin < 5 || isempty(writeReport)
    writeReport = true;
end
if nargin < 6 || isempty(finalArmMode)
    finalArmMode = 'robustNIS';
end
if nargin < 7 || isempty(adaptiveFusionOverrides)
    adaptiveFusionOverrides = struct();
end
if nargin < 8
    armSelection = [];
end

reportPath = '';
summary = struct();

%% 1. 实验固定骨架：传感器数量、通信半径、融合拓扑和基础动态权重配置
% 这里定义的是“所有 arm 共享”的默认设置。后面的 buildArms 会复制这份
% baseAdaptiveFusionConfig，再按 arm 覆盖局部字段。注意：脚本里仍保留
% 一些历史次线字段（如 freshness/NIS/history）用于旧报告兼容；当前
% computeAdaptiveFusionWeights 主线已经不再消费这些次线字段。
staggeredBirths = true;
leaderSensor = 8;
sensorCommRange = 150;
fusionWeighting = 'Metropolis';
baseAdaptiveFusionConfig = struct( ...
    'enabled', true, ...
    'emaAlpha', 0.7, ...
    'minWeight', 0.05, ...
    'useCovariance', true, ...
    'useLinkQuality', true, ...
    'useCardinalityConsensus', false, ...
    'cardinalityConsensusScale', 4.0, ...
    'cardinalityConsensusMinScore', 0.4, ...
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
    'useFreshness', false, ...
    'freshnessDecay', 0.5, ...
    'freshnessMinScore', 0.4, ...
    'useCtFiDecay', false, ...
    'ctFiDecayMinScore', 0.35, ...
    'ctFiDecayPower', 1.0, ...
    'ctFiUseDetectionProbability', true, ...
    'ctFiProcessNoiseScale', NaN, ...
    'fiTraceUseExistenceProbability', false, ...
    'fiTraceExistencePower', 1.0, ...
    'fiTraceUseDetectionProbability', false, ...
    'fiTraceUseClutterPenalty', false, ...
    'pdWeightPower', 1.0, ...
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
baseAdaptiveFusionConfig = mergeStructFields(baseAdaptiveFusionConfig, adaptiveFusionOverrides);

%% 2. 场景侧配置：传感器质量、通信丢包分档、传感器运动和目标编队出生
% 这一段只准备输入，不运行滤波。真正的模型对象在 trial 循环里生成，
% 这样每个 trial 可以用确定 seed 生成配对可比的随机场景/通信结果。
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
commConfig = mergeStructFields(commConfig, commConfigOverrides);

sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.0;
sensorMotionConfig.initialStates = buildSensorInitialStates();

targetFormationConfig = struct();
targetFormationConfig.targetFormationEnabled = true;
targetFormationConfig.targetFormationStaggeredBirths = staggeredBirths;
targetFormationConfig.targetFormationBirthInterval = 8;
targetFormationConfig.targetFormationStartTime = 1;
targetFormationConfig.targetFormationLifeSpan = 100;
targetFormationConfig.targetBirthStates = buildTargetBirthStates();
targetFormationConfig.targetFormationCount = size(targetFormationConfig.targetBirthStates, 2);

%% 3. Arm 构造：把一个 finalArmMode 展开成一组可比较方法
% 最终 paper 主线通常使用 finalArmMode='fidFiaExistenceRefinement'，
% 对应 Fixed -> FID-FIA baseline -> Balanced -> Cardinality-critical。
% armSelection 只是调试入口，可临时只跑某几个 arm，不改变默认实验定义。
arms = buildArms(baseAdaptiveFusionConfig, finalArmMode);
arms = selectArms(arms, armSelection);
armNames = {arms.name};
numArms = numel(arms);

%% 4. 预分配结果容器：local metrics、network disagreement 和 runtime
% local metrics 是每个 sensor 相对 ground truth 的 tracking 质量；
% consensus metrics 是不同 sensor 输出之间的一致性/分歧，不直接看 truth。
eOspa = zeros(numberOfTrials, numberOfSensors, numArms);
hOspa = zeros(numberOfTrials, numberOfSensors, numArms);
rmse = zeros(numberOfTrials, numberOfSensors, numArms);
cardErr = zeros(numberOfTrials, numberOfSensors, numArms);
consOspa = zeros(numberOfTrials, numArms);
consPos = zeros(numberOfTrials, numArms);
consCard = zeros(numberOfTrials, numArms);
consOspaSeries = [];
consPosSeries = [];
consCardSeries = [];
filterRuntimeSeconds = zeros(numberOfTrials, numArms);
pDropBySensorTrials = zeros(numberOfTrials, numberOfSensors);

%% 5. Trial 主循环：每个 trial 生成一次共享场景，再让所有 arm 在同一场景上配对比较
for trial = 1:numberOfTrials
    fprintf('Trial %d/%d\n', trial, numberOfTrials);
    if useFixedSeed
        rng(baseSeed + trial);
    end

    % 5.1 生成模型和测量：后续所有 arm 复用同一份 ground truth / measurements。
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

    % 5.2 通信模型：先可选多速率采样，再应用 packet-drop / bandwidth / outage。
    % computeAdaptiveFusionWeights 后面只看 delivered measurements 和 commStats，
    % 因此通信层统计是动态 linkQuality / availabilityMask 的直接输入。
    [measurementsForComm, samplingStats] = applyOptionalMultiRateSchedule(measurements, commConfig);
    [measurementsDelivered, commStats] = applyCommunicationModel(measurementsForComm, model, commConfig);
    commStats = attachSamplingStats(commStats, samplingStats);
    pDropBySensorTrials(trial, :) = reshape(commStats.pDropBySensor, 1, []);

    % 5.3 Arm 循环：每个 arm 只替换 adaptiveFusion 配置，其余场景输入保持相同。
    for armIdx = 1:numArms
        fprintf('  Arm %d/%d: %s\n', armIdx, numArms, arms(armIdx).name);
        armModel = model;
        armModel.adaptiveFusion = arms(armIdx).adaptiveFusion;
        neighborMap = buildNeighborMap4Plus4(numberOfSensors);
        runtimeStart = tic();
        % 关键调用：进入分布式滤波层。
        % runDistributedLmbFilter 会为每个 sensor 构造 local model；
        % local model 再进入 runParallelUpdateLmbFilter；
        % 动态权重真正发生在 runParallelUpdateLmbFilter -> computeAdaptiveFusionWeights；
        % GA 融合权重最终被 gaLmbTrackMerging 消费。
        [stateEstimatesBySensor, localModels] = runDistributedLmbFilter( ...
            armModel, measurementsDelivered, sensorTrajectories, neighborMap, commStats);
        filterRuntimeSeconds(trial, armIdx) = toc(runtimeStart);
        fprintf('    Filter runtime: %.3f s\n', filterRuntimeSeconds(trial, armIdx));

        % 5.4 Local tracking metrics：每个 sensor 的输出分别和 ground truth 比。
        % 这些指标用于防止“节点之间更一致，但一起偏离真值”的假改善。
        for s = 1:numberOfSensors
            [eArm, hArm, cardArm] = computeSimulationOspa(localModels{s}, groundTruthRfs, stateEstimatesBySensor{s});
            eOspa(trial, s, armIdx) = mean(eArm);
            hOspa(trial, s, armIdx) = mean(hArm);
            rmse(trial, s, armIdx) = mean(computeSetRmseOverTime(stateEstimatesBySensor{s}, groundTruthRfs), 'omitnan');
            cardErr(trial, s, armIdx) = mean(abs(cardArm - groundTruthRfs.cardinality));
        end

        % 5.5 Network disagreement metrics：比较 sensor 之间输出是否一致。
        % paper 主表里的 OSPA consensus error / matched localization disagreement /
        % cardinality dispersion 就来自这里的 consOspa / consPos / consCard。
        [posArm, cardArm, ospaArm] = computeConsensusMetrics(stateEstimatesBySensor, armModel);
        consOspa(trial, armIdx) = mean(ospaArm);
        consPos(trial, armIdx) = mean(posArm, 'omitnan');
        consCard(trial, armIdx) = mean(cardArm);
        if isempty(consOspaSeries)
            simLength = numel(ospaArm);
            consOspaSeries = zeros(numberOfTrials, simLength, numArms);
            consPosSeries = NaN(numberOfTrials, simLength, numArms);
            consCardSeries = zeros(numberOfTrials, simLength, numArms);
        end
        consOspaSeries(trial, :, armIdx) = reshape(ospaArm, 1, []);
        consPosSeries(trial, :, armIdx) = reshape(posArm, 1, []);
        consCardSeries(trial, :, armIdx) = reshape(cardArm, 1, []);
    end
end

%% 6. 控制台摘要：快速确认 arm 顺序、通信分档和三项 network disagreement
fprintf('=====================================\n');
fprintf('GA Tiered Link Ablation (N=%d)\n', numberOfTrials);
fprintf('Order: %s\n', strjoin(armNames, ' -> '));
fprintf('Tier levels=%s, counts=%s\n', mat2str(commConfig.pDropLevels, 3), mat2str(commConfig.pDropLevelCounts));
fprintf('pDropBySensor=%s\n', mat2str(mean(pDropBySensorTrials, 1), 4));
fprintf('=====================================\n');
for armIdx = 1:numArms
    fprintf('%s: OSPA %.6f, RMSE %.6f, Card %.6f, Runtime %.3f s\n', arms(armIdx).name, ...
        mean(consOspa(:, armIdx)), mean(consPos(:, armIdx), 'omitnan'), ...
        mean(consCard(:, armIdx)), mean(filterRuntimeSeconds(:, armIdx)));
end

%% 7. summary 结构：给脚本调用方/批处理脚本复用，不必解析 markdown 报告
summary.armNames = armNames;
summary.consensus.ospa = mean(consOspa, 1);
summary.consensus.pos = mean(consPos, 1, 'omitnan');
summary.consensus.card = mean(consCard, 1);
summary.consensusTrials.ospa = consOspa;
summary.consensusTrials.pos = consPos;
summary.consensusTrials.card = consCard;
summary.local.eOspa = squeeze(mean(eOspa, 1));
summary.local.hOspa = squeeze(mean(hOspa, 1));
summary.local.rmse = squeeze(mean(rmse, 1));
summary.local.cardErr = squeeze(mean(cardErr, 1));
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
summary.runtime.meanSecondsPerStep = summary.runtime.meanFilterSeconds ./ targetFormationConfig.targetFormationLifeSpan;
summary.runtime.relativeToFixed = summary.runtime.meanFilterSeconds ./ max(summary.runtime.meanFilterSeconds(1), eps);
summary.trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
if ~isempty(consOspaSeries)
    summary.consensusSeries.time = (1:size(consOspaSeries, 2))';
    summary.consensusSeries.ospa = squeeze(mean(consOspaSeries, 1));
    summary.consensusSeries.pos = squeeze(mean(consPosSeries, 1, 'omitnan'));
    summary.consensusSeries.card = squeeze(mean(consCardSeries, 1));
    summary.consensusSeries.armNames = armNames;
end
summary.pDropBySensorTrials = pDropBySensorTrials;
summary.meanPDropBySensor = mean(pDropBySensorTrials, 1);
summary.commConfig = commConfig;
summary.samplingStats = samplingStats;
summary.arms = arms;

%% 8. 可选报告写出：把配置、trial 级结果、统计量、runtime 和 local metrics 写成 markdown
if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPrefix = '';
    if any(strcmpi(finalArmMode, {'freshness', 'fresh', 'ctfidecay', 'ct_fi_decay', ...
            'informationdecay', 'information_decay', 'multiratectfi', ...
            'multi_rate_ct_fi', 'fiweightedga', 'fi_weighted_ga', ...
            'fitracega', 'fi_trace_ga', 'cardinality', 'cardinalityconsensus', ...
            'cardinality_consensus'}))
        reportPrefix = 'Del_';
    end
    reportName = sprintf('%sGA_TIERED_LINK_ABLATION_N%d_SEED%d_%s.md', ...
        reportPrefix, numberOfTrials, baseSeed, timestamp);
    reportPath = fullfile(reportDir, reportName);
    writeAblationReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
        sensorCommRange, fusionWeighting, leaderSensor, commConfig, pDropBySensorTrials, ...
        arms, consOspa, consPos, consCard, eOspa, hOspa, rmse, cardErr, ...
        filterRuntimeSeconds, targetFormationConfig.targetFormationLifeSpan, finalArmMode);
    fprintf('Report written: %s\n', reportPath);
end
end

function arms = selectArms(arms, armSelection)
% SELECTARMS - 调试用的 arm 子集选择器。
% 正常论文实验不传 armSelection，默认跑完整 arm 序列。只有 smoke test
% 或排查某个 arm 时，才用数字索引、名称片段或 'final' 只跑子集。
if nargin < 2 || isempty(armSelection)
    return;
end

if isnumeric(armSelection)
    armIdx = unique(max(1, min(numel(arms), round(armSelection(:)'))));
    arms = arms(armIdx);
    return;
end

if ischar(armSelection) || isstring(armSelection)
    requested = cellstr(armSelection);
elseif iscell(armSelection)
    requested = cellfun(@char, armSelection, 'UniformOutput', false);
else
    return;
end

matched = false(1, numel(arms));
for i = 1:numel(requested)
    query = lower(strtrim(requested{i}));
    if isempty(query)
        continue;
    end
    for armIdx = 1:numel(arms)
        if strcmpi(query, 'final') && armIdx == numel(arms)
            matched(armIdx) = true;
        elseif contains(lower(arms(armIdx).name), query)
            matched(armIdx) = true;
        end
    end
end

if any(matched)
    arms = arms(matched);
end
end

function arms = buildArms(baseAdaptiveFusionConfig, finalArmMode)
% BUILDARMS - 把 finalArmMode 映射成一组可配对比较的 adaptiveFusion 配置。
%
% 这里是实验设计最重要的 helper。主函数每个 trial 只生成一次场景，
% 然后这里构造的每个 arm 都在同一场景/通信 realization 上运行，保证
% 指标差异主要来自 adaptiveFusion 配置，而不是随机场景差异。
%
% 当前 paper 主线：
%   finalArmMode='fidFiaExistenceRefinement'
%   -> Fixed Metropolis
%   -> Cao-Zhao FID-FIA baseline
%   -> Balanced mode (+structure-aware decoupled KLA)
%   -> Cardinality-critical mode (+FID-FIA existence refinement)
arms = repmat(struct('name', '', 'adaptiveFusion', struct()), 1, 5);

% 默认前三个 arm 是旧消融骨架：fixed -> covariance -> link quality。
% 某些 finalArmMode 会在 switch 里重写整组 arms，例如当前主线的
% fidFiaExistenceRefinement 直接改成 4-arm paper-facing 顺序。
cfg = baseAdaptiveFusionConfig;
cfg.enabled = false;
cfg.useDecoupledKla = false;
cfg.useCovariance = false;
cfg.useLinkQuality = false;
cfg.useNIS = false;
arms(1).name = 'fixed weights';
arms(1).adaptiveFusion = cfg;

cfg = baseAdaptiveFusionConfig;
cfg.enabled = true;
cfg.useDecoupledKla = false;
cfg.useCovariance = true;
cfg.useLinkQuality = false;
cfg.useNIS = false;
arms(2).name = '+covariance';
arms(2).adaptiveFusion = cfg;

cfg = baseAdaptiveFusionConfig;
cfg.enabled = true;
cfg.useDecoupledKla = false;
cfg.useCovariance = true;
cfg.useLinkQuality = true;
cfg.useNIS = false;
arms(3).name = '+link quality';
arms(3).adaptiveFusion = cfg;

switch lower(finalArmMode)
    case {'fidfiaexistencerefinement', 'fid_fia_existence_refinement', ...
            'fid-fia-existence-refinement', 'caozhaoexistencerefinement', ...
            'cao_zhao_existence_refinement'}
        % Paper 主实验 arm 顺序。
        % 1. fixed weights：关闭 adaptiveFusion，保留 Metropolis 拓扑权重。
        % 2. FID-FIA baseline：整个 posterior 用 scalar FID-FIA 权重融合。
        % 3. Balanced：三因子 backbone + decoupled KLA + structure prior。
        % 4. Cardinality-critical：在 Balanced 基础上，只给 existence 分支加 FID-FIA。
        arms = repmat(struct('name', '', 'adaptiveFusion', struct()), 1, 4);

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = false;
        cfg.method = 'factorized';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useNIS = false;
        cfg.useFidFiaExistence = false;
        arms(1).name = 'fixed weights';
        arms(1).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'fidFia';
        cfg.useDecoupledKla = false;
        cfg.useStructureAwareKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useExistenceConfidence = false;
        cfg.useFreshness = false;
        cfg.useHistory = false;
        cfg.useNIS = false;
        cfg.useFidFiaExistence = false;
        cfg.fidFiaUseEma = false;
        cfg.fidFiaMinWeight = 0.0;
        cfg.fidFiaUseExistenceWeight = true;
        cfg.fidFiaExistencePower = 1.0;
        cfg.fidFiaQuadraturePoints = 3;
        cfg.fidFiaUseDetectionProbability = true;
        arms(2).name = 'Cao-Zhao FID-FIA baseline';
        arms(2).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'factorized';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        cfg.useDecoupledKla = true;
        cfg.useStructureAwareKla = true;
        cfg.usePosteriorStructureConsistency = false;
        cfg.useFidFiaExistence = false;
        if abs(cfg.existenceConfidenceMinScore - 0.6) < 1e-9
            cfg.existenceConfidenceMinScore = 0.85;
        end
        if abs(cfg.existenceConfidencePower - 1.0) < 1e-9
            cfg.existenceConfidencePower = 2.0;
        end
        if abs(cfg.spatialDecouplingStrength - 1.0) < 1e-9
            cfg.spatialDecouplingStrength = 0.5;
        end
        if abs(cfg.existenceDecouplingStrength - 1.0) < 1e-9
            cfg.existenceDecouplingStrength = 0.15;
        end
        if cfg.spatialStructureStrength <= 0
            cfg.spatialStructureStrength = 0.45;
        end
        if cfg.existenceStructureStrength <= 0
            cfg.existenceStructureStrength = 0.08;
        end
        if cfg.structureReliabilityPower <= 0
            cfg.structureReliabilityPower = 0.30;
        end
        arms(3).name = '+structure-aware decoupled KLA';
        arms(3).adaptiveFusion = cfg;

        % Cardinality-critical 的关键不是“替换 Balanced”，而是在 Balanced 的
        % existence branch 上叠加 FID-FIA。这里把 FID-FIA score floor 和
        % existence final weight floor 都设成 0，让目标数风险高时可以强抑制
        % 不可靠 existence 分支；spatial branch 的 0.05 floor 不变。
        cfg.useFidFiaExistence = true;
        cfg.fidFiaExistenceStrength = 4.0;
        cfg.fidFiaExistenceMinScore = 0.0;
        cfg.existenceEmaAlpha = 0.0;
        cfg.existenceMinWeight = 0.0;
        arms(4).name = '+FID-FIA existence refinement';
        arms(4).adaptiveFusion = cfg;
    case {'fidfia', 'fid_fia', 'fidfiabaseline', 'fid_fia_baseline', ...
            'fisherfia', 'fisher_fia', 'informationgeometry', ...
            'information_geometry', 'caozhao', 'cao_zhao'}
        arms = repmat(struct('name', '', 'adaptiveFusion', struct()), 1, 3);

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = false;
        cfg.method = 'factorized';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useNIS = false;
        arms(1).name = 'fixed weights';
        arms(1).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'fidFia';
        cfg.useDecoupledKla = false;
        cfg.useStructureAwareKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useExistenceConfidence = false;
        cfg.useFreshness = false;
        cfg.useHistory = false;
        cfg.useNIS = false;
        cfg.fidFiaUseEma = false;
        cfg.fidFiaMinWeight = 0.0;
        cfg.fidFiaUseExistenceWeight = true;
        cfg.fidFiaExistencePower = 1.0;
        cfg.fidFiaQuadraturePoints = 3;
        cfg.fidFiaUseDetectionProbability = true;
        arms(2).name = 'Cao-Zhao FID-FIA baseline';
        arms(2).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'factorized';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        cfg.useDecoupledKla = true;
        cfg.useStructureAwareKla = true;
        cfg.usePosteriorStructureConsistency = false;
        if abs(cfg.existenceConfidenceMinScore - 0.6) < 1e-9
            cfg.existenceConfidenceMinScore = 0.85;
        end
        if abs(cfg.existenceConfidencePower - 1.0) < 1e-9
            cfg.existenceConfidencePower = 2.0;
        end
        if abs(cfg.spatialDecouplingStrength - 1.0) < 1e-9
            cfg.spatialDecouplingStrength = 0.5;
        end
        if abs(cfg.existenceDecouplingStrength - 1.0) < 1e-9
            cfg.existenceDecouplingStrength = 0.15;
        end
        if cfg.spatialStructureStrength <= 0
            cfg.spatialStructureStrength = 0.45;
        end
        if cfg.existenceStructureStrength <= 0
            cfg.existenceStructureStrength = 0.08;
        end
        if cfg.structureReliabilityPower <= 0
            cfg.structureReliabilityPower = 0.30;
        end
        arms(3).name = '+structure-aware decoupled KLA';
        arms(3).adaptiveFusion = cfg;
    case {'freshness', 'fresh'}
        % 历史次线入口：当前 computeAdaptiveFusionWeights 主线已不再消费
        % freshness 分数。保留这个分支主要是为了旧报告/历史脚本可读。
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useFreshness = true;
        arms(4).name = '+freshness';
        arms(4).adaptiveFusion = cfg;
        arms = arms(1:4);
    case {'ctfidecay', 'ct_fi_decay', 'informationdecay', 'information_decay', ...
            'multiratectfi', 'multi_rate_ct_fi'}
        % 历史多速率/信息衰减入口。当前动态权重核心已清理 CT-FI decay，
        % 不再把它作为 paper 主线动态权重因子。
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = true;
        cfg.useCtFiDecay = false;
        cfg.useNIS = false;
        arms(4).name = '+freshness';
        arms(4).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useCtFiDecay = true;
        cfg.useNIS = false;
        arms(5).name = '+CT-FI information decay';
        arms(5).adaptiveFusion = cfg;
    case {'fiweightedga', 'fi_weighted_ga', 'fitracega', 'fi_trace_ga', ...
            'fisherweightedga', 'fisher_weighted_ga'}
        arms = repmat(struct('name', '', 'adaptiveFusion', struct()), 1, 4);

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = false;
        cfg.method = 'factorized';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        arms(1).name = 'fixed weights';
        arms(1).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'pdWeightedGa';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        arms(2).name = 'PD-weighted GA';
        arms(2).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'fiTraceGa';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = false;
        cfg.useLinkQuality = false;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        arms(3).name = 'FI-weighted GA';
        arms(3).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.method = 'factorized';
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        arms(4).name = '+link quality';
        arms(4).adaptiveFusion = cfg;
    case {'cardinality', 'cardinalityconsensus', 'cardinality_consensus'}
        % 历史 cardinality-consensus 尝试。当前核心函数已不再消费该分数。
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useCardinalityConsensus = true;
        arms(4).name = '+cardinality consensus';
        arms(4).adaptiveFusion = cfg;
        arms = arms(1:4);
    case {'existence', 'existenceconfidence', 'existence_confidence'}
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        arms(4).name = '+existence confidence';
        arms(4).adaptiveFusion = cfg;
        arms = arms(1:4);
    case {'decoupled', 'decoupledkla', 'decoupled_kla'}
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        arms(4).name = '+existence confidence';
        arms(4).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        cfg.useDecoupledKla = true;
        arms(5).name = '+decoupled KLA';
        arms(5).adaptiveFusion = cfg;
    case {'structureaware', 'structure_aware', 'structure-aware', ...
            'structureawaredecoupledkla', 'structure_aware_decoupled_kla', ...
            'structure-aware-decoupled-kla'}
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        arms(4).name = '+existence confidence';
        arms(4).adaptiveFusion = cfg;

        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = false;
        cfg.useExistenceConfidence = true;
        cfg.useDecoupledKla = true;
        cfg.useStructureAwareKla = true;
        cfg.usePosteriorStructureConsistency = false;
        if abs(cfg.existenceConfidenceMinScore - 0.6) < 1e-9
            cfg.existenceConfidenceMinScore = 0.85;
        end
        if abs(cfg.existenceConfidencePower - 1.0) < 1e-9
            cfg.existenceConfidencePower = 2.0;
        end
        if abs(cfg.spatialDecouplingStrength - 1.0) < 1e-9
            cfg.spatialDecouplingStrength = 0.5;
        end
        if abs(cfg.existenceDecouplingStrength - 1.0) < 1e-9
            cfg.existenceDecouplingStrength = 0.15;
        end
        if cfg.spatialStructureStrength <= 0
            cfg.spatialStructureStrength = 0.45;
        end
        if cfg.existenceStructureStrength <= 0
            cfg.existenceStructureStrength = 0.08;
        end
        if cfg.structureReliabilityPower <= 0
            cfg.structureReliabilityPower = 0.30;
        end
        arms(5).name = '+structure-aware decoupled KLA';
        arms(5).adaptiveFusion = cfg;
    otherwise
        % 旧默认 robust-NIS 消融入口。保留是为了旧脚本兼容；当前 paper
        % 主线请显式传 finalArmMode='fidFiaExistenceRefinement'。
        cfg = baseAdaptiveFusionConfig;
        cfg.enabled = true;
        cfg.useDecoupledKla = false;
        cfg.useCovariance = true;
        cfg.useLinkQuality = true;
        cfg.useFreshness = false;
        cfg.useNIS = true;
        cfg.robustNIS = true;
        arms(4).name = '+robust NIS';
        arms(4).adaptiveFusion = cfg;
        arms = arms(1:4);
end
end

function writeAblationReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, ...
    sensorCommRange, fusionWeighting, leaderSensor, commConfig, pDropBySensorTrials, ...
    arms, consOspa, consPos, consCard, eOspa, hOspa, rmse, cardErr, ...
    filterRuntimeSeconds, simulationLength, finalArmMode)
% WRITEABLATIONREPORT - 把一次实验的配置和结果落成 markdown 报告。
%
% 报告分三层：
%   1. Run Config / Arm Configs：复现实验需要的参数；
%   2. Network disagreement / local tracking / runtime：paper 表格用的统计量；
%   3. Paired improvement：同一 trial 内和 baseline 成对比较，减少随机性影响。

fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
baselineName = arms(1).name;
fprintf(fid, '# GA Tiered Link Ablation (%s)\n\n', timestamp);
fprintf(fid, 'Comparison order: %s\n\n', strjoin({arms.name}, ' -> '));
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n', baseSeed, useFixedSeed);
if useFixedSeed
    fprintf(fid, '- trialSeeds: %s\n', mat2str(computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed)));
end
fprintf(fid, '- sensorCommRange: %d\n', sensorCommRange);
fprintf(fid, '- fusionWeighting: %s\n', fusionWeighting);
fprintf(fid, '- leaderSensor: %d\n', leaderSensor);
fprintf(fid, '- linkModel: %s\n', getField(commConfig, 'linkModel', 'fixed'));
fprintf(fid, '- pDrop target mean: %.3f\n', getField(commConfig, 'pDrop', 0));
fprintf(fid, '- pDropLevels: %s\n', mat2str(getField(commConfig, 'pDropLevels', []), 3));
fprintf(fid, '- pDropLevelCounts: %s\n', mat2str(getField(commConfig, 'pDropLevelCounts', [])));
if getField(commConfig, 'enableMultiRate', false)
    fprintf(fid, '- samplingPeriods: %s\n', mat2str(getField(commConfig, 'samplingPeriods', [])));
    fprintf(fid, '- samplingPhaseOffsets: %s\n', mat2str(getField(commConfig, 'samplingPhaseOffsets', [])));
end
fprintf(fid, '\n');
fprintf(fid, '- finalArmMode: %s\n\n', finalArmMode);

fprintf(fid, '## Arm Configs\n');
for armIdx = 1:numel(arms)
    cfg = arms(armIdx).adaptiveFusion;
    fprintf(fid, '### %s\n', arms(armIdx).name);
    fprintf(fid, '- enabled: %d\n', getField(cfg, 'enabled', false));
    fprintf(fid, '- method: %s\n', char(getField(cfg, 'method', 'factorized')));
    fprintf(fid, '- useCovariance: %d\n', getField(cfg, 'useCovariance', false));
    fprintf(fid, '- useLinkQuality: %d\n', getField(cfg, 'useLinkQuality', false));
    fprintf(fid, '- useExistenceConfidence: %d\n', getField(cfg, 'useExistenceConfidence', false));
    fprintf(fid, '- useHistorySmoothedExistenceConfidence: %d\n', getField(cfg, 'useHistorySmoothedExistenceConfidence', false));
    fprintf(fid, '- existenceHistoryEmaAlpha: %.3f\n', getField(cfg, 'existenceHistoryEmaAlpha', 0.8));
    fprintf(fid, '- useFreshness: %d\n', getField(cfg, 'useFreshness', false));
    fprintf(fid, '- useCtFiDecay: %d\n', getField(cfg, 'useCtFiDecay', false));
    if getField(cfg, 'useCtFiDecay', false)
        fprintf(fid, '- ctFiDecayMinScore: %.3f\n', getField(cfg, 'ctFiDecayMinScore', 0.35));
        fprintf(fid, '- ctFiDecayPower: %.3f\n', getField(cfg, 'ctFiDecayPower', 1.0));
        fprintf(fid, '- ctFiUseDetectionProbability: %d\n', getField(cfg, 'ctFiUseDetectionProbability', true));
        fprintf(fid, '- ctFiProcessNoiseScale: %.3f\n', getField(cfg, 'ctFiProcessNoiseScale', NaN));
    end
    methodName = char(getField(cfg, 'method', 'factorized'));
    if strcmpi(methodName, 'fiTraceGa')
        fprintf(fid, '- fiTraceUseExistenceProbability: %d\n', getField(cfg, 'fiTraceUseExistenceProbability', false));
        fprintf(fid, '- fiTraceExistencePower: %.3f\n', getField(cfg, 'fiTraceExistencePower', 1.0));
        fprintf(fid, '- fiTraceUseDetectionProbability: %d\n', getField(cfg, 'fiTraceUseDetectionProbability', false));
        fprintf(fid, '- fiTraceUseClutterPenalty: %d\n', getField(cfg, 'fiTraceUseClutterPenalty', false));
    elseif strcmpi(methodName, 'pdWeightedGa')
        fprintf(fid, '- pdWeightPower: %.3f\n', getField(cfg, 'pdWeightPower', 1.0));
    end
    fprintf(fid, '- useNIS: %d\n', getField(cfg, 'useNIS', false));
    fprintf(fid, '- useDecoupledKla: %d\n', getField(cfg, 'useDecoupledKla', false));
    fprintf(fid, '- useStructureAwareKla: %d\n', getField(cfg, 'useStructureAwareKla', false));
    fprintf(fid, '- usePosteriorStructureConsistency: %d\n', getField(cfg, 'usePosteriorStructureConsistency', false));
    fprintf(fid, '- existenceConfidenceMinScore: %.3f\n', getField(cfg, 'existenceConfidenceMinScore', 0));
    fprintf(fid, '- existenceConfidencePower: %.3f\n', getField(cfg, 'existenceConfidencePower', 0));
    fprintf(fid, '- spatialEmaAlpha: %.3f\n', getField(cfg, 'spatialEmaAlpha', 0));
    fprintf(fid, '- existenceEmaAlpha: %.3f\n', getField(cfg, 'existenceEmaAlpha', 0));
    fprintf(fid, '- spatialMinWeight: %.3f\n', getField(cfg, 'spatialMinWeight', 0));
    fprintf(fid, '- existenceMinWeight: %.3f\n', getField(cfg, 'existenceMinWeight', 0));
    fprintf(fid, '- spatialDecouplingStrength: %.3f\n', getField(cfg, 'spatialDecouplingStrength', 0));
    fprintf(fid, '- existenceDecouplingStrength: %.3f\n', getField(cfg, 'existenceDecouplingStrength', 0));
    fprintf(fid, '- spatialStructureStrength: %.3f\n', getField(cfg, 'spatialStructureStrength', 0));
    fprintf(fid, '- existenceStructureStrength: %.3f\n', getField(cfg, 'existenceStructureStrength', 0));
    fprintf(fid, '- structureReliabilityPower: %.3f\n', getField(cfg, 'structureReliabilityPower', 0));
    fprintf(fid, '- structureReliabilityMinScore: %.3f\n', getField(cfg, 'structureReliabilityMinScore', 0));
    fprintf(fid, '- useFidFiaExistence: %d\n', getField(cfg, 'useFidFiaExistence', false));
    printFidFiaConfig = strcmpi(getField(cfg, 'method', 'factorized'), 'fidFia') || ...
        getField(cfg, 'useFidFiaExistence', false);
    if printFidFiaConfig
        fprintf(fid, '- fidFiaExistenceStrength: %.3f\n', getField(cfg, 'fidFiaExistenceStrength', 0.5));
        fprintf(fid, '- fidFiaExistenceMinScore: %.3f\n', getField(cfg, 'fidFiaExistenceMinScore', 0.4));
        fprintf(fid, '- fidFiaUseExistenceWeight: %d\n', getField(cfg, 'fidFiaUseExistenceWeight', true));
        fprintf(fid, '- fidFiaExistencePower: %.3f\n', getField(cfg, 'fidFiaExistencePower', 1.0));
        fprintf(fid, '- fidFiaQuadraturePoints: %d\n', getField(cfg, 'fidFiaQuadraturePoints', 3));
        fprintf(fid, '- fidFiaUseDetectionProbability: %d\n', getField(cfg, 'fidFiaUseDetectionProbability', true));
        fprintf(fid, '- fidFiaUseEma: %d\n', getField(cfg, 'fidFiaUseEma', false));
        fprintf(fid, '- fidFiaMinWeight: %.3f\n\n', getField(cfg, 'fidFiaMinWeight', 0));
    else
        fprintf(fid, '\n');
    end
end

fprintf(fid, '## Per-Trial pDropBySensor\n');
for trial = 1:size(pDropBySensorTrials, 1)
    fprintf(fid, '- Trial %d: %s\n', trial, mat2str(pDropBySensorTrials(trial, :), 4));
end
fprintf(fid, '\n');

fprintf(fid, '## Per-Trial Network Disagreement Metrics\n');
fprintf(fid, '| Trial | Seed | Arm | OSPA | RMSE | Cardinality |\n');
fprintf(fid, '|------:|-----:|:----|-----:|-----:|------------:|\n');
trialSeeds = computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed);
for trial = 1:numberOfTrials
    for armIdx = 1:numel(arms)
        fprintf(fid, '| %d | %.0f | %s | %.6f | %.6f | %.6f |\n', ...
            trial, trialSeeds(trial), arms(armIdx).name, ...
            consOspa(trial, armIdx), consPos(trial, armIdx), consCard(trial, armIdx));
    end
end
fprintf(fid, '\n');

fprintf(fid, '## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '| Arm | OSPA | RMSE | Cardinality |\n');
fprintf(fid, '|:----|-----:|-----:|------------:|\n');
for armIdx = 1:numel(arms)
    fprintf(fid, '| %s | %.6f | %.6f | %.6f |\n', arms(armIdx).name, ...
        mean(consOspa(:, armIdx)), mean(consPos(:, armIdx), 'omitnan'), mean(consCard(:, armIdx)));
end

fprintf(fid, '\n## Network Disagreement Metrics With Trial Variability\n');
writeMetricStatsTable(fid, {arms.name}, {'OSPA', 'RMSE', 'Cardinality'}, ...
    {consOspa, consPos, consCard}, [false, true, false]);

if numel(arms) >= 2
    fprintf(fid, '\n## Paired Improvements Relative to %s\n', baselineName);
    writePairedImprovementTable(fid, {arms.name}, {'OSPA', 'RMSE', 'Cardinality'}, ...
        {consOspa, consPos, consCard}, [false, true, false]);
end

fprintf(fid, '\n## Computational Cost\n');
fprintf(fid, 'Filter wall-clock time measures only the distributed LMB filtering/fusion call for each arm. Scenario generation, communication-model sampling, and metric evaluation are excluded.\n\n');
writeRuntimeCostSummaryTable(fid, {arms.name}, filterRuntimeSeconds, simulationLength, baselineName);
fprintf(fid, '\n');
writeRuntimePerTrialTable(fid, computeTrialSeeds(numberOfTrials, baseSeed, useFixedSeed), ...
    {arms.name}, filterRuntimeSeconds, simulationLength, baselineName);

fprintf(fid, '\n## Local Tracking Metrics (mean across sensors and trials)\n');
fprintf(fid, '| Arm | E-OSPA | RMSE | CardErr |\n');
fprintf(fid, '|:----|-------:|-----:|--------:|\n');
for armIdx = 1:numel(arms)
    eOspaMean = computeGlobalMean(eOspa(:, :, armIdx), false);
    rmseMean = computeGlobalMean(rmse(:, :, armIdx), true);
    cardErrMean = computeGlobalMean(cardErr(:, :, armIdx), false);
    fprintf(fid, '| %s | %.6f | %.6f | %.6f |\n', arms(armIdx).name, ...
        eOspaMean, rmseMean, cardErrMean);
end

fprintf(fid, '\n## Local Tracking Metrics With Trial Variability\n');
localEOspaTrial = computeTrialSensorMeans(eOspa, false);
localRmseTrial = computeTrialSensorMeans(rmse, true);
localCardTrial = computeTrialSensorMeans(cardErr, false);
writeMetricStatsTable(fid, {arms.name}, {'E-OSPA', 'RMSE', 'CardErr'}, ...
    {localEOspaTrial, localRmseTrial, localCardTrial}, [false, true, false]);

if numel(arms) >= 2
    fprintf(fid, '\n## Paired Local-Metric Improvements Relative to %s\n', baselineName);
    writePairedImprovementTable(fid, {arms.name}, {'E-OSPA', 'RMSE', 'CardErr'}, ...
        {localEOspaTrial, localRmseTrial, localCardTrial}, [false, true, false]);
end

fprintf(fid, '\n## Local Tracking Metrics By Sensor (mean across trials)\n');
fprintf(fid, '| Sensor | Arm | E-OSPA | RMSE | CardErr |\n');
fprintf(fid, '|------:|:----|-------:|-----:|--------:|\n');
for sensorIdx = 1:size(eOspa, 2)
    for armIdx = 1:numel(arms)
        fprintf(fid, '| %d | %s | %.6f | %.6f | %.6f |\n', sensorIdx, arms(armIdx).name, ...
            mean(eOspa(:, sensorIdx, armIdx)), ...
            mean(rmse(:, sensorIdx, armIdx), 'omitnan'), mean(cardErr(:, sensorIdx, armIdx)));
    end
end

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
baselineIdx = 1;
for metricIdx = 1:numel(metricNames)
    data = metricArrays{metricIdx};
    omitnanFlag = omitnanFlags(metricIdx);
    baseline = data(:, baselineIdx);
    for armIdx = 2:numel(armNames)
        candidate = data(:, armIdx);
        valid = isfinite(baseline) & isfinite(candidate);
        if ~omitnanFlag
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
if nargin < 4 || isempty(simulationLength) || simulationLength <= 0
    simulationLength = 1;
end
if nargin < 5 || isempty(baselineName)
    baselineName = armNames{1};
end
if isempty(runtimeSeconds)
    fprintf(fid, 'Runtime data unavailable.\n');
    return;
end

fixedRuntime = runtimeSeconds(:, 1);
fprintf(fid, '| Arm | Filter runtime (s) | Runtime/step (s) | Relative to %s | N |\n', baselineName);
fprintf(fid, '|:----|-------------------:|-----------------:|------------------:|--:|\n');
for armIdx = 1:numel(armNames)
    values = runtimeSeconds(:, armIdx);
    stats = summarizeVector(values, true);
    validRatio = isfinite(values) & isfinite(fixedRuntime) & fixedRuntime > eps;
    if armIdx == 1
        ratioMean = 1.0;
    elseif any(validRatio)
        ratioStats = summarizeVector(values(validRatio) ./ fixedRuntime(validRatio), true);
        ratioMean = ratioStats.mean;
    else
        ratioMean = NaN;
    end
    fprintf(fid, '| %s | %.6f +/- %.6f | %.6f | %.3fx | %d |\n', ...
        armNames{armIdx}, stats.mean, stats.std, stats.mean / simulationLength, ...
        ratioMean, stats.n);
end

if numel(armNames) >= 2
    fprintf(fid, '\n');
    fprintf(fid, '| Arm | Paired overhead (s) | Relative overhead | Slower trials |\n');
    fprintf(fid, '|:----|--------------------:|------------------:|--------------:|\n');
    for armIdx = 2:numel(armNames)
        values = runtimeSeconds(:, armIdx);
        valid = isfinite(values) & isfinite(fixedRuntime) & fixedRuntime > eps;
        deltas = values(valid) - fixedRuntime(valid);
        ratios = values(valid) ./ fixedRuntime(valid);
        deltaStats = summarizeVector(deltas, true);
        ratioStats = summarizeVector(ratios, true);
        fprintf(fid, '| %s | %.6f +/- %.6f | %.2f%% | %d/%d |\n', ...
            armNames{armIdx}, deltaStats.mean, deltaStats.std, ...
            100 * (ratioStats.mean - 1), sum(deltas > 0), deltaStats.n);
    end
end
end

function writeRuntimePerTrialTable(fid, trialSeeds, armNames, runtimeSeconds, simulationLength, baselineName)
if nargin < 5 || isempty(simulationLength) || simulationLength <= 0
    simulationLength = 1;
end
if nargin < 6 || isempty(baselineName)
    baselineName = armNames{1};
end
if isempty(runtimeSeconds)
    return;
end

fprintf(fid, '### Per-Trial Filter Runtime\n');
fprintf(fid, '| Trial | Seed | Arm | Runtime (s) | Runtime/step (s) | Relative to %s |\n', baselineName);
fprintf(fid, '|------:|-----:|:----|------------:|-----------------:|------------------:|\n');
for trial = 1:size(runtimeSeconds, 1)
    fixedRuntime = runtimeSeconds(trial, 1);
    for armIdx = 1:numel(armNames)
        runtimeValue = runtimeSeconds(trial, armIdx);
        relativeValue = runtimeValue / max(fixedRuntime, eps);
        fprintf(fid, '| %d | %.0f | %s | %.6f | %.6f | %.3fx |\n', ...
            trial, trialSeeds(trial), armNames{armIdx}, runtimeValue, ...
            runtimeValue / simulationLength, relativeValue);
    end
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
if ndims(arrayLike) < 3
    values = computeGlobalMean(arrayLike, omitnanFlag);
    return;
end
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
% COMPUTECONSENSUSMETRICS - 计算跨节点 disagreement，而不是 truth error。
%
% posConsensus  : 对每对 sensor 的估计集合做 Hungarian matching 后求位置 RMSE；
% cardConsensus : 每个 sensor 的目标数和全网 median count 的平均偏差；
% ospaConsensus : 对每对 sensor 输出做对称 OSPA，衡量有限集层面的 disagreement。
%
% 这些指标回答的是“各节点融合后是否趋于一致”。因此报告里必须同时
% 保留 local tracking metrics，防止只优化一致性而牺牲 truth-referenced 质量。
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
