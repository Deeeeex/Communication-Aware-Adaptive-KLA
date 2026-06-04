function [gaWeights, aaWeights, debug] = computeAdaptiveFusionWeights(measurementUpdatedDistributions, measurements, model, t, commStats, prevWeights)
% COMPUTEADAPTIVEFUSIONWEIGHTS - Compute adaptive GA/AA fusion weights.
%   [gaWeights, aaWeights, debug] = computeAdaptiveFusionWeights(measurementUpdatedDistributions, measurements, model, t, commStats, prevWeights)
%
%   The paper-facing factorized model is
%       mask * covariance * link * existenceConfidence
%   followed by optional branch-decoupled KLA, weak structure-aware modulation,
%   and the Cardinality-critical FID-FIA refinement on the existence branch.
%   文件导读：
%       GA/AA 动态权重的核心实现。入口接收每个传感器的 local posterior、
%       measurements、model 配置、当前时刻 t、通信/诊断统计和上一时刻权重。
%       当前论文主线只保留 covariance、realized link quality、
%       existence confidence、branch-decoupled KLA、structure-aware prior
%       以及 FID-FIA existence refinement。freshness、NIS、history 等弱证据
%       扩展已从这个核心函数中移除，避免阅读时把历史尝试误认为主方法。
%       输出包括 GA/AA scalar weights、spatial/existence branch weights、
%       target-wise weights，以及用于报告和排错的 debug 因子。

numSensors = model.numberOfSensors;

%% 1. 读取配置：决定当前是 factorized 主线还是 PD/FI/FID-FIA direct baseline
cfg = struct();
if isfield(model, 'adaptiveFusion') && isstruct(model.adaptiveFusion)
    cfg = model.adaptiveFusion;
end

method = lower(getField(cfg, 'method', 'factorized'));
emaAlpha = getField(cfg, 'emaAlpha', 0.7);
% final weight floor：作用在归一化后的权重上，不改变单个 score 的含义。
% 主实验通常用 0.05，目的是让活跃邻居保留少量贡献，避免某一步的
% 瞬时 score 抖动把节点彻底清零。Cardinality-critical 的 existence 分支
% 会把 existenceMinWeight 设为 0，让 FID-FIA 可以强力压低不可靠的基数分支。
minWeight = getField(cfg, 'minWeight', 0.0);
useDecoupledKla = getField(cfg, 'useDecoupledKla', false);
useCovariance = getField(cfg, 'useCovariance', true);
useLinkQuality = getField(cfg, 'useLinkQuality', true);
useExistenceConfidence = getField(cfg, 'useExistenceConfidence', false);
spatialEmaAlpha = getField(cfg, 'spatialEmaAlpha', emaAlpha);
existenceEmaAlpha = getField(cfg, 'existenceEmaAlpha', emaAlpha);
spatialMinWeight = getField(cfg, 'spatialMinWeight', minWeight);
existenceMinWeight = getField(cfg, 'existenceMinWeight', minWeight);
spatialCovariancePower = max(getField(cfg, 'spatialCovariancePower', 1.0), 0);
spatialLinkQualityPower = max(getField(cfg, 'spatialLinkQualityPower', 1.0), 0);
existenceLinkQualityPower = max(getField(cfg, 'existenceLinkQualityPower', 1.0), 0);
existenceConfidenceWeightPower = max(getField(cfg, 'existenceConfidenceWeightPower', 1.0), 0);
spatialDecouplingStrength = min(max(getField(cfg, 'spatialDecouplingStrength', 1.0), 0), 1);
existenceDecouplingStrength = min(max(getField(cfg, 'existenceDecouplingStrength', 1.0), 0), 1);
spatialStructureStrength = max(getField(cfg, 'spatialStructureStrength', 0.0), 0);
existenceStructureStrength = max(getField(cfg, 'existenceStructureStrength', 0.0), 0);
structureReliabilityPower = max(getField(cfg, 'structureReliabilityPower', 0.0), 0);
% score floor：作用在某个质量因子的映射内部。这里的通信可靠性下界
% 防止结构先验因为丢包率过高而完全消灭一个仍在邻域里的传感器。
structureReliabilityMinScore = min(max(getField(cfg, 'structureReliabilityMinScore', 0.25), 0), 1);
useStructureAwareKla = getField(cfg, 'useStructureAwareKla', false) || ...
    spatialStructureStrength > 0 || existenceStructureStrength > 0;
usePosteriorStructureConsistency = getField(cfg, 'usePosteriorStructureConsistency', true);
useFidFiaExistence = getField(cfg, 'useFidFiaExistence', false);

availabilityMask = resolveAvailabilityMask(model, commStats, t, numSensors);
%% 2. Direct baseline：这些方法直接给出权重，不进入 factor product 主线
if isFidFiaMethod(method)
    [gaWeights, aaWeights, debug] = computeFidFiaFusionWeights( ...
        measurementUpdatedDistributions, model, t, cfg, availabilityMask, prevWeights);
    return;
end
if isFiTraceGaMethod(method)
    [gaWeights, aaWeights, debug] = computeFiTraceGaFusionWeights( ...
        measurementUpdatedDistributions, model, t, cfg, availabilityMask);
    return;
end
if isPdWeightedGaMethod(method)
    [gaWeights, aaWeights, debug] = computePdWeightedGaFusionWeights( ...
        measurementUpdatedDistributions, model, t, cfg, availabilityMask);
    return;
end

% 先把每个传感器的 local posterior 压缩成三个主线分数：
%   covScore                  : posterior 越集中越大；没有显式下界。
%   linkQuality               : 当前步 delivered/(delivered+dropped)；没有显式下界。
%   existenceConfidenceScore  : existence probability 越果断越大；有 score 下界。
% availabilityMask 是硬门控，和 score floor 不同：不可用邻居会直接被置零。
covScore = computeCovarianceScore(measurementUpdatedDistributions, model);
[existenceConfidenceScore, existenceHistoryState, existenceConfidenceInstantScore] = ...
    resolveExistenceConfidenceScore( ...
        measurementUpdatedDistributions, useExistenceConfidence, cfg, prevWeights);
linkQuality = computeLinkQuality(measurements, commStats, t, numSensors);
[covScore, linkQuality] = applyFactorMasks(covScore, linkQuality, useCovariance, useLinkQuality);
expectedCardinality = computeExpectedCardinality(measurementUpdatedDistributions);
[fidFiaExistenceScore, fidFiaScore, fidFiaPairCounts] = resolveFidFiaExistenceScore( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask, useFidFiaExistence);

%% 3. Factorized backbone：只保留论文主线采用的三类质量信号
% baseScore 先把“能不能用”和“空间/链路质量”合起来；rawScore 再乘
% existence confidence，补上基数/存在性维度。这里不用 score 下界去保护
% covScore/linkQuality，是为了让空 posterior 或完全未送达的传感器自然降权；
% 只有 existence confidence 保留下界，因为“存在性不够果断”通常应弱化，
% 但不应单独把空间上仍有价值的传感器完全打掉。
baseScore = availabilityMask .* covScore .* linkQuality;
rawScore = baseScore .* existenceConfidenceScore;

if useDecoupledKla
    %% 4. Decoupled KLA：空间分支和存在/基数分支使用不同的专用 score
    % rawScore 是两条分支共同的 anchor，避免两条路径完全脱节。
    % dedicatedScore 则表达“这一条分支更关心什么”：
    %   spatial    : covScore + linkQuality，主要服务 Gaussian spatial fusion。
    %   existence  : existenceConfidence + linkQuality，主要服务 Bernoulli r fusion。
    % blendDecoupledScore 用几何插值，而不是线性加法，这样弱分数仍然会
    % 对最终权重产生约束，不会被另一个大分数简单抵消。
    spatialDedicatedScore = availabilityMask .* (covScore .^ spatialCovariancePower) .* ...
        (linkQuality .^ spatialLinkQualityPower);
    existenceDedicatedScore = availabilityMask .* (linkQuality .^ existenceLinkQualityPower) .* ...
        (existenceConfidenceScore .^ existenceConfidenceWeightPower);
    spatialScore = blendDecoupledScore(rawScore, spatialDedicatedScore, spatialDecouplingStrength);
    existenceScore = blendDecoupledScore(rawScore, existenceDedicatedScore, existenceDecouplingStrength);
    spatialStructurePrior = resolveStructurePrior(model, 'gaSpatialStructurePrior', 'gaTopologyWeights', numSensors);
    existenceStructurePrior = resolveStructurePrior(model, 'gaExistenceStructurePrior', 'gaTopologyWeights', numSensors);
    communicationReliabilityPrior = ones(1, numSensors);
    if structureReliabilityPower > 0
        communicationReliabilityPrior = resolveCommunicationReliabilityPrior( ...
            commStats, numSensors, structureReliabilityMinScore);
        spatialStructurePrior = applyStructurePrior( ...
            spatialStructurePrior, communicationReliabilityPrior, structureReliabilityPower);
        existenceStructurePrior = applyStructurePrior( ...
            existenceStructurePrior, communicationReliabilityPrior, structureReliabilityPower);
    end
    spatialStructureScore = ones(1, numSensors);
    existenceStructureScore = ones(1, numSensors);
    %% 4.1 可选 structure-aware 调制：用拓扑/结构一致性轻量修正两条分支
    % 当前主实验采用 usePosteriorStructureConsistency=false：结构项来自邻域
    % 先验和通信可靠性 prior，而不是重新计算 posterior pairwise disagreement。
    % 如果打开 posterior consistency，则下面的 helper 会再用 spatial/existence
    % consistency min score 给结构一致性分数设置下界，避免结构项变成硬剔除。
    if useStructureAwareKla
        if usePosteriorStructureConsistency
            [spatialStructureScore, existenceStructureScore] = resolveStructureConsistencyScores( ...
                measurementUpdatedDistributions, model, spatialStructurePrior, existenceStructurePrior, cfg);
            spatialScore = spatialScore .* (spatialStructureScore .^ spatialStructureStrength);
            existenceScore = existenceScore .* (existenceStructureScore .^ existenceStructureStrength);
        else
            spatialScore = applyStructurePrior(spatialScore, spatialStructurePrior, spatialStructureStrength);
            existenceScore = applyStructurePrior(existenceScore, existenceStructurePrior, existenceStructureStrength);
            spatialStructureScore = spatialStructurePrior;
            existenceStructureScore = existenceStructurePrior;
        end
    end
    %% 4.2 Cardinality-critical refinement：FID-FIA 只调制 existence branch
    % FID-FIA 衡量同一传感器下目标对的可分辨性，更直接影响“有几个目标”
    % 和 existence/cardinality 判定。它不进入 spatialScore，是为了保留
    % Balanced mode 的空间定位优势。最终 Cardinality-critical arm 里，
    % fidFiaExistenceMinScore=0 且 existenceMinWeight=0，因此 FID-FIA 可以
    % 对 existence branch 做强抑制；spatial branch 仍保留 0.05 权重下界。
    if useFidFiaExistence
        fidFiaExistenceStrength = max(getField(cfg, 'fidFiaExistenceStrength', 0.5), 0);
        existenceScore = existenceScore .* (fidFiaExistenceScore .^ fidFiaExistenceStrength);
    end

    %% 4.3 归一化、EMA 平滑和最小权重保护
    % finalizeAdaptiveWeights 的顺序是 normalize -> EMA -> normalize -> floor。
    % floor 放在最后，是因为它保护的是最终融合权重，而不是中间 score。
    % 这也解释了为什么 score 下界和 weight 下界不能互相替代。
    spatialPrev = resolvePreviousWeights(prevWeights, 'gaSpatial', 'ga', numSensors);
    existencePrev = resolvePreviousWeights(prevWeights, 'gaExistence', 'ga', numSensors);

    spatialWeights = finalizeAdaptiveWeights(spatialScore, availabilityMask, spatialPrev, ...
        spatialEmaAlpha, spatialMinWeight);
    existenceWeights = finalizeAdaptiveWeights(existenceScore, availabilityMask, existencePrev, ...
        existenceEmaAlpha, existenceMinWeight);

    gaWeights = spatialWeights;
    aaWeights = spatialWeights;
    rawWeights = spatialWeights;
else
    %% 5. 非解耦路径：所有因子合成一条 scalar 权重，同时用于 spatial/existence
    % 这个路径主要用于 factorized baseline。没有 spatial/existence 分支差异，
    % 所以 minWeight 只作为一条统一权重的最终下界。
    rawWeights = normalizeScores(rawScore, availabilityMask);

    weights = rawWeights;
    if nargin >= 6 && isstruct(prevWeights) && isfield(prevWeights, 'ga')
        if numel(prevWeights.ga) == numSensors
            weights = emaAlpha * prevWeights.ga + (1 - emaAlpha) * rawWeights;
            weights = normalizeScores(weights, availabilityMask);
        end
    end

    if minWeight > 0
        weights = enforceMinimumWeight(weights, availabilityMask, minWeight);
    end

    gaWeights = weights;
    aaWeights = weights;
    spatialWeights = weights;
    existenceWeights = weights;
end

%% 6. debug 输出：保留主线因子和最终权重，供报告、消融和排错使用
debug = struct();
debug.availabilityMask = availabilityMask;
debug.covScore = covScore;
debug.baseScore = baseScore;
debug.existenceConfidenceScore = existenceConfidenceScore;
debug.existenceConfidenceInstantScore = existenceConfidenceInstantScore;
debug.linkQuality = linkQuality;
debug.rawScore = rawScore;
debug.rawWeights = rawWeights;
debug.weights = gaWeights;
debug.useDecoupledKla = useDecoupledKla;
debug.useStructureAwareKla = useStructureAwareKla;
debug.usePosteriorStructureConsistency = usePosteriorStructureConsistency;
debug.useFidFiaExistence = useFidFiaExistence;
debug.fidFiaExistenceScore = fidFiaExistenceScore;
debug.fidFiaScore = fidFiaScore;
debug.fidFiaPairCounts = fidFiaPairCounts;
debug.spatialRawScore = rawScore;
debug.existenceRawScore = rawScore;
debug.spatialStructurePrior = ones(1, numSensors);
debug.existenceStructurePrior = ones(1, numSensors);
debug.communicationReliabilityPrior = ones(1, numSensors);
debug.spatialStructureScore = ones(1, numSensors);
debug.existenceStructureScore = ones(1, numSensors);
if useDecoupledKla
    debug.spatialRawScore = spatialScore;
    debug.existenceRawScore = existenceScore;
    debug.spatialStructurePrior = spatialStructurePrior;
    debug.existenceStructurePrior = existenceStructurePrior;
    debug.communicationReliabilityPrior = communicationReliabilityPrior;
    debug.spatialStructureScore = spatialStructureScore;
    debug.existenceStructureScore = existenceStructureScore;
end
debug.gaSpatialWeights = spatialWeights;
debug.aaSpatialWeights = spatialWeights;
debug.gaExistenceWeights = existenceWeights;
debug.aaExistenceWeights = existenceWeights;
debug.expectedCardinality = expectedCardinality;
if ~isempty(fieldnames(existenceHistoryState))
    debug.historyState = existenceHistoryState;
end
end

function tf = isFidFiaMethod(method)
tf = any(strcmpi(method, {'fidfia', 'fid_fia', 'fisherfia', ...
    'fisher_fia', 'informationgeometry', 'information_geometry', ...
    'caozhao', 'cao_zhao'}));
end

function tf = isFiTraceGaMethod(method)
tf = any(strcmpi(method, {'fitracega', 'fi_trace_ga', 'fi-weighted-ga', ...
    'fiweightedga', 'fi_weighted_ga', 'fishertracega', ...
    'fisher_trace_ga'}));
end

function tf = isPdWeightedGaMethod(method)
tf = any(strcmpi(method, {'pdweightedga', 'pd_weighted_ga', ...
    'pd-weighted-ga', 'detectionweightedga', 'detection_weighted_ga'}));
end

function [gaWeights, aaWeights, debug] = computeFiTraceGaFusionWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask)

numSensors = numel(measurementUpdatedDistributions);
[targetWeights, sensorScores, targetScores] = computeTargetWiseFiTraceWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask);
weights = normalizeScores(sensorScores, availabilityMask);

gaWeights = weights;
aaWeights = weights;
debug = buildDirectWeightDebug( ...
    measurementUpdatedDistributions, availabilityMask, sensorScores, weights, 'fiTraceGa');
debug.fiTraceScore = sensorScores;
debug.fiTraceTargetScores = targetScores;
debug.gaTargetWiseWeights = targetWeights;
debug.aaTargetWiseWeights = targetWeights;
debug.gaSpatialWeights = weights;
debug.aaSpatialWeights = weights;
debug.gaExistenceWeights = weights;
debug.aaExistenceWeights = weights;
debug.useTargetWiseFiWeights = true;
debug.numberOfSensors = numSensors;
end

function [gaWeights, aaWeights, debug] = computePdWeightedGaFusionWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask)

[targetWeights, sensorScores, targetScores] = computeTargetWisePdWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask);
weights = normalizeScores(sensorScores, availabilityMask);

gaWeights = weights;
aaWeights = weights;
debug = buildDirectWeightDebug( ...
    measurementUpdatedDistributions, availabilityMask, sensorScores, weights, 'pdWeightedGa');
debug.pdScore = sensorScores;
debug.pdTargetScores = targetScores;
debug.gaTargetWiseWeights = targetWeights;
debug.aaTargetWiseWeights = targetWeights;
debug.gaSpatialWeights = weights;
debug.aaSpatialWeights = weights;
debug.gaExistenceWeights = weights;
debug.aaExistenceWeights = weights;
debug.useTargetWiseFiWeights = false;
end

function debug = buildDirectWeightDebug(measurementUpdatedDistributions, availabilityMask, rawScore, weights, methodName)
numSensors = numel(rawScore);
debug = struct();
debug.availabilityMask = availabilityMask;
debug.covScore = ones(1, numSensors);
debug.baseScore = rawScore;
debug.existenceConfidenceScore = ones(1, numSensors);
debug.linkQuality = ones(1, numSensors);
debug.rawScore = rawScore;
debug.rawWeights = weights;
debug.weights = weights;
debug.method = methodName;
debug.useFidFiaExistence = false;
debug.fidFiaScore = ones(1, numSensors);
debug.fidFiaExistenceScore = ones(1, numSensors);
debug.fidFiaPairCounts = zeros(1, numSensors);
debug.useDecoupledKla = false;
debug.useStructureAwareKla = false;
debug.usePosteriorStructureConsistency = false;
debug.spatialRawScore = rawScore;
debug.existenceRawScore = rawScore;
debug.spatialStructurePrior = ones(1, numSensors);
debug.existenceStructurePrior = ones(1, numSensors);
debug.communicationReliabilityPrior = ones(1, numSensors);
debug.spatialStructureScore = ones(1, numSensors);
debug.existenceStructureScore = ones(1, numSensors);
debug.expectedCardinality = computeExpectedCardinality(measurementUpdatedDistributions);
end

function [targetWeights, sensorScores, targetScores] = computeTargetWiseFiTraceWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask)

numSensors = numel(measurementUpdatedDistributions);
numTargets = resolveNumberOfBernoulliTracks(measurementUpdatedDistributions);
targetScores = zeros(numTargets, numSensors);
for i = 1:numTargets
    for s = 1:numSensors
        if availabilityMask(s) <= 0
            continue;
        end
        targetScores(i, s) = estimateBernoulliFiTrace( ...
            measurementUpdatedDistributions{s}, i, model, s, t, cfg);
    end
end

targetWeights = normalizeTargetScores(targetScores, availabilityMask);
sensorScores = summarizeTargetScores(targetScores, availabilityMask);
end

function [targetWeights, sensorScores, targetScores] = computeTargetWisePdWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask)

numSensors = numel(measurementUpdatedDistributions);
numTargets = resolveNumberOfBernoulliTracks(measurementUpdatedDistributions);
targetScores = zeros(numTargets, numSensors);
for i = 1:numTargets
    for s = 1:numSensors
        if availabilityMask(s) <= 0
            continue;
        end
        targetScores(i, s) = estimateBernoulliDetectionScore( ...
            measurementUpdatedDistributions{s}, i, model, s, t, cfg);
    end
end

targetWeights = normalizeTargetScores(targetScores, availabilityMask);
sensorScores = summarizeTargetScores(targetScores, availabilityMask);
end

function numTargets = resolveNumberOfBernoulliTracks(measurementUpdatedDistributions)
numTargets = 0;
for s = 1:numel(measurementUpdatedDistributions)
    numTargets = max(numTargets, numel(measurementUpdatedDistributions{s}));
end
end

function score = estimateBernoulliFiTrace(objects, objectIdx, model, sensorIdx, t, cfg)
score = 0;
if objectIdx > numel(objects) || objects(objectIdx).numberOfGmComponents < 1
    return;
end

[mu, covariance] = mprojection(model.xDimension, objects(objectIdx));
covariance = regularizeCovariance(covariance);
infoTrace = trace(pinv(covariance));
if ~isfinite(infoTrace) || infoTrace <= 0
    return;
end

scale = 1;
if getField(cfg, 'fiTraceUseExistenceProbability', false)
    existencePower = max(getField(cfg, 'fiTraceExistencePower', 1.0), 0);
    scale = scale * (max(objects(objectIdx).r, 0) ^ existencePower);
end
if getField(cfg, 'fiTraceUseDetectionProbability', false)
    [pdSensor, ~] = evaluateSensorQuality(model, sensorIdx, mu, t);
    scale = scale * max(pdSensor, 0);
end
if getField(cfg, 'fiTraceUseClutterPenalty', false)
    clutterRate = resolveSensorValue(model.clutterRate, sensorIdx, 0);
    scale = scale / (1 + max(clutterRate, 0));
end

score = max(scale, 0) * infoTrace;
end

function score = estimateBernoulliDetectionScore(objects, objectIdx, model, sensorIdx, t, cfg)
score = 0;
if objectIdx > numel(objects) || objects(objectIdx).numberOfGmComponents < 1
    return;
end
[mu, ~] = mprojection(model.xDimension, objects(objectIdx));
[pdSensor, ~] = evaluateSensorQuality(model, sensorIdx, mu, t);
score = max(pdSensor, 0) ^ max(getField(cfg, 'pdWeightPower', 1.0), 0);
end

function targetWeights = normalizeTargetScores(targetScores, availabilityMask)
targetWeights = zeros(size(targetScores));
for i = 1:size(targetScores, 1)
    targetWeights(i, :) = normalizeScores(targetScores(i, :), availabilityMask);
end
end

function sensorScores = summarizeTargetScores(targetScores, availabilityMask)
if isempty(targetScores)
    sensorScores = availabilityMask;
    return;
end
sensorScores = mean(targetScores, 1);
sensorScores(~isfinite(sensorScores)) = 0;
if ~any(sensorScores .* availabilityMask > 0)
    sensorScores = availabilityMask;
end
end

function [gaWeights, aaWeights, debug] = computeFidFiaFusionWeights( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask, prevWeights)

numSensors = numel(measurementUpdatedDistributions);
[fiaScore, fidPairCounts] = computeFidFiaScore(measurementUpdatedDistributions, model, t, cfg);
rawWeights = normalizeScores(fiaScore, availabilityMask);

weights = rawWeights;
useEma = getField(cfg, 'fidFiaUseEma', false);
if useEma && nargin >= 6 && isstruct(prevWeights) && isfield(prevWeights, 'ga') && ...
        numel(prevWeights.ga) == numSensors
    emaAlpha = getField(cfg, 'fidFiaEmaAlpha', getField(cfg, 'emaAlpha', 0.7));
    weights = emaAlpha * prevWeights.ga + (1 - emaAlpha) * rawWeights;
    weights = normalizeScores(weights, availabilityMask);
end

minWeight = getField(cfg, 'fidFiaMinWeight', 0.0);
if minWeight > 0
    weights = enforceMinimumWeight(weights, availabilityMask, minWeight);
end
% 纯 FID-FIA baseline 在主实验中通常 fidFiaMinWeight=0，目的是忠实呈现
% scalar FID-FIA 权重本身，而不是用 final weight floor 稀释这个 baseline。

gaWeights = weights;
aaWeights = weights;

debug = struct();
debug.availabilityMask = availabilityMask;
debug.covScore = ones(1, numSensors);
debug.baseScore = fiaScore;
debug.existenceConfidenceScore = ones(1, numSensors);
debug.linkQuality = ones(1, numSensors);
debug.rawScore = fiaScore;
debug.rawWeights = rawWeights;
debug.weights = weights;
debug.method = 'fidFia';
debug.fiaScore = fiaScore;
debug.fidPairCounts = fidPairCounts;
debug.useFidFiaExistence = false;
debug.fidFiaScore = rawWeights;
debug.fidFiaExistenceScore = ones(1, numSensors);
debug.fidFiaPairCounts = fidPairCounts;
debug.useDecoupledKla = false;
debug.useStructureAwareKla = false;
debug.usePosteriorStructureConsistency = false;
debug.spatialRawScore = fiaScore;
debug.existenceRawScore = fiaScore;
debug.spatialStructurePrior = ones(1, numSensors);
debug.existenceStructurePrior = ones(1, numSensors);
debug.communicationReliabilityPrior = ones(1, numSensors);
debug.spatialStructureScore = ones(1, numSensors);
debug.existenceStructureScore = ones(1, numSensors);
debug.gaSpatialWeights = weights;
debug.aaSpatialWeights = weights;
debug.gaExistenceWeights = weights;
debug.aaExistenceWeights = weights;
debug.expectedCardinality = computeExpectedCardinality(measurementUpdatedDistributions);
end

function [fidFiaExistenceScore, normalizedFiaScore, fidPairCounts] = resolveFidFiaExistenceScore( ...
    measurementUpdatedDistributions, model, t, cfg, availabilityMask, useFidFiaExistence)

numSensors = numel(measurementUpdatedDistributions);
fidFiaExistenceScore = ones(1, numSensors);
normalizedFiaScore = zeros(1, numSensors);
fidPairCounts = zeros(1, numSensors);
if ~useFidFiaExistence
    return;
end

[fiaScore, fidPairCounts] = computeFidFiaScore(measurementUpdatedDistributions, model, t, cfg);
availableScores = fiaScore(:)' .* reshape(availabilityMask, 1, []);
maxScore = max(availableScores);
if isfinite(maxScore) && maxScore > eps
    normalizedFiaScore = availableScores / maxScore;
end
normalizedFiaScore(~isfinite(normalizedFiaScore)) = 0;
normalizedFiaScore = min(max(normalizedFiaScore, 0), 1);

% FID-FIA existence score 的下界是方法选择的一部分：
%   default 0.4  : 作为温和 refinement 时，不让 FID-FIA 单独否决一个节点。
%   final arm 0 : Cardinality-critical 模式下允许强抑制 existence branch。
% 这里先得到 score，下游是否仍给最终权重留 floor 由 existenceMinWeight 决定。
scorePower = max(getField(cfg, 'fidFiaExistencePower', 1.0), 0);
minScore = min(max(getField(cfg, 'fidFiaExistenceMinScore', 0.4), 0), 1);
fidFiaExistenceScore = minScore + (1 - minScore) * (normalizedFiaScore .^ scorePower);
end

function [fiaScore, fidPairCounts] = computeFidFiaScore(measurementUpdatedDistributions, model, t, cfg)
numSensors = numel(measurementUpdatedDistributions);
fiaScore = zeros(1, numSensors);
fidPairCounts = zeros(1, numSensors);

existenceThreshold = getField(cfg, 'fidFiaExistenceThreshold', getField(model, 'existenceThreshold', 0));
useExistenceWeight = getField(cfg, 'fidFiaUseExistenceWeight', true);
existencePower = max(getField(cfg, 'fidFiaExistencePower', 1.0), 0);

for s = 1:numSensors
    objects = measurementUpdatedDistributions{s};
    if isempty(objects)
        continue;
    end

    [states, existenceProb] = extractFidFiaTargetStates(objects, model, existenceThreshold);
    numTargets = size(states, 2);
    if numTargets < 2
        continue;
    end

    for i = 1:(numTargets - 1)
        for j = (i + 1):numTargets
            pairWeight = 1;
            if useExistenceWeight
                pairWeight = (max(existenceProb(i), 0) * max(existenceProb(j), 0)) ^ existencePower;
            end
            if pairWeight <= 0
                continue;
            end

            fidValue = approximateLinearGaussianFid(model, s, states(:, i), states(:, j), t, cfg);
            if isfinite(fidValue) && fidValue > 0
                fiaScore(s) = fiaScore(s) + pairWeight * fidValue;
                fidPairCounts(s) = fidPairCounts(s) + 1;
            end
        end
    end
end
end

function [states, existenceProb] = extractFidFiaTargetStates(objects, model, existenceThreshold)
states = zeros(model.xDimension, numel(objects));
existenceProb = zeros(1, numel(objects));
count = 0;
for i = 1:numel(objects)
    if objects(i).numberOfGmComponents < 1 || objects(i).r <= existenceThreshold
        continue;
    end
    [mu, ~] = mprojection(model.xDimension, objects(i));
    count = count + 1;
    states(:, count) = mu;
    existenceProb(count) = objects(i).r;
end
states = states(:, 1:count);
existenceProb = existenceProb(1:count);
end

function fidValue = approximateLinearGaussianFid(model, sensorIdx, stateA, stateB, t, cfg)
fidValue = 0;
if isempty(stateA) || isempty(stateB) || numel(stateA) < 2 || numel(stateB) < 2
    return;
end

deltaState = stateB(:) - stateA(:);
deltaPos = deltaState(1:2);
if norm(deltaPos) <= eps
    return;
end

numPoints = max(1, round(getField(cfg, 'fidFiaQuadraturePoints', 3)));
useDetectionProbability = getField(cfg, 'fidFiaUseDetectionProbability', true);
Hpos = resolvePositionMeasurementJacobian(model, sensorIdx);
if isempty(Hpos)
    return;
end

if numPoints == 1
    lambdas = 0.5;
else
    lambdas = linspace(0, 1, numPoints);
end
integrand = zeros(size(lambdas));

for idx = 1:numel(lambdas)
    state = stateA(:) + lambdas(idx) * deltaState;
    [pdSensor, measurementCovariance] = evaluateSensorQuality(model, sensorIdx, state, t);
    if useDetectionProbability
        if pdSensor <= 0
            integrand(idx) = 0;
            continue;
        end
        detectionScale = sqrt(max(pdSensor, 0));
    else
        detectionScale = 1;
    end

    measurementCovariance = regularizeCovariance(measurementCovariance);
    fisherMetric = Hpos' * (measurementCovariance \ Hpos);
    metricDistanceSquared = deltaPos' * fisherMetric * deltaPos;
    integrand(idx) = detectionScale * sqrt(max(real(metricDistanceSquared), 0));
end

if numPoints == 1
    fidValue = integrand(1);
else
    fidValue = trapz(lambdas, integrand);
end
end

function Hpos = resolvePositionMeasurementJacobian(model, sensorIdx)
Hpos = [];
if ~isfield(model, 'C') || numel(model.C) < sensorIdx || isempty(model.C{sensorIdx})
    return;
end
C = model.C{sensorIdx};
if size(C, 2) < 2
    return;
end
Hpos = C(:, 1:2);
end

function covariance = regularizeCovariance(covariance)
covariance = (covariance + covariance') / 2;
if isempty(covariance)
    return;
end
if rcond(covariance) < 1e-12
    covariance = covariance + 1e-9 * eye(size(covariance));
end
end

function prior = resolveStructurePrior(model, preferredField, fallbackField, numSensors)
prior = ones(1, numSensors);
if nargin < 1 || ~isstruct(model)
    return;
end
if isfield(model, preferredField) && numel(model.(preferredField)) == numSensors
    prior = reshape(model.(preferredField), 1, []);
elseif isfield(model, fallbackField) && numel(model.(fallbackField)) == numSensors
    prior = reshape(model.(fallbackField), 1, []);
end
prior = max(prior, eps);
prior = prior / mean(prior);
end

function adjustedScore = applyStructurePrior(score, prior, strength)
adjustedScore = score;
if nargin < 3 || strength <= 0 || isempty(prior)
    return;
end
adjustedScore = score .* (prior .^ strength);
end

function [spatialScore, existenceScore] = resolveStructureConsistencyScores( ...
    measurementUpdatedDistributions, model, spatialPrior, existencePrior, cfg)

numSensors = numel(measurementUpdatedDistributions);
spatialScore = ones(1, numSensors);
existenceScore = ones(1, numSensors);
if numSensors <= 1
    return;
end

spatialScale = max(getField(cfg, 'spatialConsistencyScale', 0.6), 0);
existenceScale = max(getField(cfg, 'existenceConsistencyScale', 2.0), 0);
% 这两个 min score 只在 usePosteriorStructureConsistency=true 时生效。
% 它们把“结构不一致”限制成软惩罚，避免 posterior 结构噪声把一个节点
% 直接置零。当前主实验关闭该分支，因此这两个下界不是 headline 设置。
spatialMinScore = min(max(getField(cfg, 'spatialConsistencyMinScore', 0.4), 0), 1);
existenceMinScore = min(max(getField(cfg, 'existenceConsistencyMinScore', 0.4), 0), 1);
summaries = repmat(struct( ...
    'r', [], ...
    'position', zeros(2, 0), ...
    'trace', [], ...
    'center', zeros(2, 1), ...
    'dispersion', 0), 1, numSensors);
for s = 1:numSensors
    summaries(s) = buildStructureSummary(measurementUpdatedDistributions{s}, model);
end

for s = 1:numSensors
    spatialWeights = reshape(spatialPrior, 1, []);
    existenceWeights = reshape(existencePrior, 1, []);
    spatialWeights(s) = 0;
    existenceWeights(s) = 0;

    spatialWeightSum = sum(spatialWeights);
    existenceWeightSum = sum(existenceWeights);
    spatialDisagreement = 0;
    existenceDisagreement = 0;

    for j = 1:numSensors
        if j == s
            continue;
        end
        [pairSpatial, pairExistence] = computePairwiseStructureDisagreement( ...
            summaries(s), summaries(j));
        spatialDisagreement = spatialDisagreement + spatialWeights(j) * pairSpatial;
        existenceDisagreement = existenceDisagreement + existenceWeights(j) * pairExistence;
    end

    if spatialWeightSum > 0
        spatialDisagreement = spatialDisagreement / spatialWeightSum;
        spatialScore(s) = spatialMinScore + (1 - spatialMinScore) * ...
            exp(-spatialScale * spatialDisagreement);
    end
    if existenceWeightSum > 0
        existenceDisagreement = existenceDisagreement / existenceWeightSum;
        existenceScore(s) = existenceMinScore + (1 - existenceMinScore) * ...
            exp(-existenceScale * existenceDisagreement);
    end
end
end

function summary = buildStructureSummary(objects, model)
summary = struct('r', [], 'position', zeros(2, 0), 'trace', [], 'center', zeros(2, 1), 'dispersion', 0);
if isempty(objects) || nargin < 2 || ~isstruct(model) || ~isfield(model, 'xDimension')
    return;
end

numObjects = numel(objects);
summary.r = extractExistenceVector(objects);
summary.position = zeros(2, numObjects);
summary.trace = zeros(1, numObjects);
for idx = 1:numObjects
    if objects(idx).numberOfGmComponents < 1
        continue;
    end
    [mu, cov] = mprojection(model.xDimension, objects(idx));
    posDim = min(2, numel(mu));
    if posDim > 0
        summary.position(1:posDim, idx) = mu(1:posDim);
    end
    summary.trace(idx) = trace(cov);
end

weights = max(summary.r, 0);
activeMask = (weights > 0) & (summary.trace > 0);
if any(activeMask)
    activeWeights = weights(activeMask);
    activePositions = summary.position(:, activeMask);
    activeTrace = summary.trace(activeMask);
    totalWeight = sum(activeWeights);
    summary.center = activePositions * (activeWeights(:) / max(totalWeight, eps));
    centeredPositions = activePositions - summary.center;
    radialSpread = sum(centeredPositions .^ 2, 1);
    summary.dispersion = sum(activeWeights .* (radialSpread + activeTrace)) / max(totalWeight, eps);
end
end

function [spatialDisagreement, existenceDisagreement] = computePairwiseStructureDisagreement(summaryA, summaryB)
spatialDisagreement = 0;
existenceDisagreement = 0;
rA = summaryA.r;
rB = summaryB.r;
maxObjects = max(numel(rA), numel(rB));
if maxObjects == 0
    return;
end

rA = padVector(rA, maxObjects);
rB = padVector(rB, maxObjects);
traceA = padVector(summaryA.trace, maxObjects);
traceB = padVector(summaryB.trace, maxObjects);

if summaryA.dispersion > 0 && summaryB.dispersion > 0
    centerDelta = summaryA.center - summaryB.center;
    spatialScale = 1 + summaryA.dispersion + summaryB.dispersion;
    centerMismatch = log(1 + (centerDelta' * centerDelta) / max(spatialScale, eps));
    spreadMismatch = abs(log((summaryA.dispersion + eps) / (summaryB.dispersion + eps)));
    spatialDisagreement = centerMismatch + 0.35 * spreadMismatch;
end

profileDiff = mean(abs(rA - rB));
expectedCardA = sum(rA);
expectedCardB = sum(rB);
expectedCardNorm = 1 + 0.5 * (expectedCardA + expectedCardB);
expectedCardDiff = abs(expectedCardA - expectedCardB) / max(expectedCardNorm, eps);
confidenceDiff = abs(computeExistenceConfidence(rA) - computeExistenceConfidence(rB));
existenceDisagreement = 0.6 * profileDiff + 0.3 * expectedCardDiff + 0.1 * confidenceDiff;
end

function confidence = computeExistenceConfidence(existenceProb)
if isempty(existenceProb)
    confidence = 0;
    return;
end
certainty = abs(2 * existenceProb - 1);
confidence = sum(existenceProb .* certainty) / (eps + sum(existenceProb));
confidence = min(max(confidence, 0), 1);
end

function values = extractExistenceVector(objects)
if isempty(objects)
    values = [];
    return;
end
values = reshape([objects.r], 1, []);
end

function padded = padVector(values, targetLength)
padded = zeros(1, targetLength);
if isempty(values)
    return;
end
count = min(numel(values), targetLength);
padded(1:count) = reshape(values(1:count), 1, []);
end

function padded = padMatrix(values, rowCount, targetColumns)
padded = zeros(rowCount, targetColumns);
if isempty(values)
    return;
end
copyRows = min(size(values, 1), rowCount);
copyCols = min(size(values, 2), targetColumns);
padded(1:copyRows, 1:copyCols) = values(1:copyRows, 1:copyCols);
end

function prior = resolveCommunicationReliabilityPrior(commStats, numSensors, minScore)
prior = ones(1, numSensors);
if nargin < 1 || ~isstruct(commStats) || ~isfield(commStats, 'pDropBySensor')
    return;
end
if numel(commStats.pDropBySensor) ~= numSensors
    return;
end
reliability = 1 - reshape(commStats.pDropBySensor, 1, []);
reliability = min(max(reliability, 0), 1);
% pDrop 只作为结构 prior 的软调制：minScore 让高丢包节点仍可保留
% 一点结构贡献，随后再除以均值，保证 prior 只改变相对比例而不整体放大/缩小。
prior = minScore + (1 - minScore) * reliability;
prior = prior / mean(prior);
end

function prev = resolvePreviousWeights(prevWeights, preferredField, fallbackField, numSensors)
prev = [];
if nargin < 1 || ~isstruct(prevWeights)
    return;
end
if isfield(prevWeights, preferredField) && numel(prevWeights.(preferredField)) == numSensors
    prev = prevWeights.(preferredField);
    return;
end
if isfield(prevWeights, fallbackField) && numel(prevWeights.(fallbackField)) == numSensors
    prev = prevWeights.(fallbackField);
end
end

function weights = finalizeAdaptiveWeights(score, mask, prev, emaAlpha, minWeight)
weights = normalizeScores(score, mask);
if ~isempty(prev)
    weights = emaAlpha * prev + (1 - emaAlpha) * weights;
    weights = normalizeScores(weights, mask);
end
if minWeight > 0
    weights = enforceMinimumWeight(weights, mask, minWeight);
end
end

function blendedScore = blendDecoupledScore(anchorScore, dedicatedScore, strength)
if strength <= 0
    blendedScore = anchorScore;
    return;
end
if strength >= 1
    blendedScore = dedicatedScore;
    return;
end
blendedScore = (anchorScore .^ (1 - strength)) .* (dedicatedScore .^ strength);
end

function [existenceConfidenceScore, historyState, instantScore] = resolveExistenceConfidenceScore( ...
    measurementUpdatedDistributions, useExistenceConfidence, cfg, prevWeights)
numSensors = numel(measurementUpdatedDistributions);
existenceConfidenceScore = ones(1, numSensors);
instantScore = ones(1, numSensors);
historyState = struct();
if ~useExistenceConfidence
    return;
end

% existence confidence 是 score floor 最关键的使用点。r 接近 0.5 说明
% 存在性判断不果断，但这并不等价于空间 posterior 完全无用。因此默认
% 通过 minScore 做温和降权；主实验 Balanced/Cardinality-critical 会把
% 0.6 的默认值提升到 0.85，让这个因子更像轻量校正而不是硬筛选。
minScore = min(max(getField(cfg, 'existenceConfidenceMinScore', 0.6), 0), 1);
power = max(getField(cfg, 'existenceConfidencePower', 1.0), 0);
useHistorySmoothedExistence = getField(cfg, 'useHistorySmoothedExistenceConfidence', false);
historyAlpha = min(max(getField(cfg, 'existenceHistoryEmaAlpha', 0.8), 0), 1);
previousHistory = resolvePreviousExistenceHistory(prevWeights, numSensors);
if useHistorySmoothedExistence
    historyState.existenceConfidence = cell(1, numSensors);
end

for s = 1:numSensors
    objects = measurementUpdatedDistributions{s};
    if isempty(objects)
        existenceConfidenceScore(s) = 1;
        instantScore(s) = 1;
        if useHistorySmoothedExistence
            historyState.existenceConfidence{s} = buildExistenceHistoryEntry(objects, []);
        end
        continue;
    end

    existenceProb = [objects.r];
    if isempty(existenceProb)
        existenceConfidenceScore(s) = 1;
        instantScore(s) = 1;
        if useHistorySmoothedExistence
            historyState.existenceConfidence{s} = buildExistenceHistoryEntry(objects, []);
        end
        continue;
    end

    instantConfidence = computeExistenceConfidenceValue(existenceProb);
    instantScore(s) = minScore + (1 - minScore) * (instantConfidence ^ power);

    scoreExistenceProb = existenceProb;
    if useHistorySmoothedExistence
        scoreExistenceProb = smoothExistenceProbabilities( ...
            objects, existenceProb, previousHistory{s}, historyAlpha);
        historyState.existenceConfidence{s} = buildExistenceHistoryEntry(objects, scoreExistenceProb);
    end

    weightedConfidence = computeExistenceConfidenceValue(scoreExistenceProb);
    existenceConfidenceScore(s) = minScore + (1 - minScore) * (weightedConfidence ^ power);
end
end

function previousHistory = resolvePreviousExistenceHistory(prevWeights, numSensors)
previousHistory = cell(1, numSensors);
if nargin < 1 || ~isstruct(prevWeights) || ~isfield(prevWeights, 'historyState') || ...
        ~isstruct(prevWeights.historyState) || ~isfield(prevWeights.historyState, 'existenceConfidence')
    return;
end
storedHistory = prevWeights.historyState.existenceConfidence;
if ~iscell(storedHistory)
    return;
end
for s = 1:min(numSensors, numel(storedHistory))
    previousHistory{s} = storedHistory{s};
end
end

function smoothedProb = smoothExistenceProbabilities(objects, existenceProb, previousEntry, alpha)
smoothedProb = reshape(existenceProb, 1, []);
if isempty(objects) || isempty(previousEntry) || ~isstruct(previousEntry) || ...
        ~isfield(previousEntry, 'labels') || ~isfield(previousEntry, 'r')
    return;
end

currentLabels = extractObjectLabels(objects);
previousLabels = previousEntry.labels;
previousR = reshape(previousEntry.r, 1, []);
if isempty(currentLabels) || isempty(previousLabels) || size(previousLabels, 2) ~= numel(previousR)
    return;
end

for idx = 1:size(currentLabels, 2)
    matchIdx = find(previousLabels(1, :) == currentLabels(1, idx) & ...
        previousLabels(2, :) == currentLabels(2, idx), 1);
    if ~isempty(matchIdx)
        smoothedProb(idx) = alpha * previousR(matchIdx) + (1 - alpha) * smoothedProb(idx);
    end
end
smoothedProb = min(max(smoothedProb, 0), 1);
end

function entry = buildExistenceHistoryEntry(objects, existenceProb)
entry = struct();
entry.labels = extractObjectLabels(objects);
entry.r = reshape(existenceProb, 1, []);
end

function labels = extractObjectLabels(objects)
labels = zeros(2, numel(objects));
for idx = 1:numel(objects)
    if isfield(objects, 'birthTime')
        labels(1, idx) = objects(idx).birthTime;
    else
        labels(1, idx) = idx;
    end
    if isfield(objects, 'birthLocation')
        labels(2, idx) = objects(idx).birthLocation;
    else
        labels(2, idx) = idx;
    end
end
end

function confidence = computeExistenceConfidenceValue(existenceProb)
if isempty(existenceProb)
    confidence = 0;
    return;
end
certainty = abs(2 * existenceProb - 1);
confidence = sum(existenceProb .* certainty) / (eps + sum(existenceProb));
confidence = min(max(confidence, 0), 1);
end

function [covScore, linkQuality] = applyFactorMasks(covScore, linkQuality, useCovariance, useLinkQuality)
if ~useCovariance
    covScore = ones(size(covScore));
end
if ~useLinkQuality
    linkQuality = ones(size(linkQuality));
end
end

function covScore = computeCovarianceScore(measurementUpdatedDistributions, model)
numSensors = numel(measurementUpdatedDistributions);
covScore = zeros(1, numSensors);
for s = 1:numSensors
    objects = measurementUpdatedDistributions{s};
    if isempty(objects)
        covScore(s) = 0;
        continue;
    end
    traceValues = zeros(1, numel(objects));
    traceCount = 0;
    for i = 1:numel(objects)
        if objects(i).numberOfGmComponents < 1
            continue;
        end
        [~, T] = mprojection(model.xDimension, objects(i));
        traceCount = traceCount + 1;
        traceValues(traceCount) = trace(T);
    end
    if traceCount == 0
        covScore(s) = 0;
    else
        meanTrace = mean(traceValues(1:traceCount));
        % covScore 不设 score floor：posterior 越集中权重越大，空 posterior
        % 或无有效 Gaussian component 可以自然变成 0，由 normalizeScores 兜底。
        covScore(s) = 1 / (eps + meanTrace);
    end
end
end

function value = resolveSensorValue(values, sensorIdx, defaultValue)
if isempty(values)
    value = defaultValue;
elseif isscalar(values)
    value = values;
elseif numel(values) >= sensorIdx
    value = values(sensorIdx);
else
    value = values(1);
end
end

function linkQuality = computeLinkQuality(measurements, commStats, t, numSensors)
linkQuality = ones(1, numSensors);
if nargin < 2 || ~isstruct(commStats)
    return;
end
hasLinkFields = isfield(commStats, 'droppedByBandwidth') && ...
    isfield(commStats, 'droppedByLink') && isfield(commStats, 'droppedByOutage');
if ~hasLinkFields || t > size(commStats.droppedByBandwidth, 2)
    return;
end
for s = 1:numSensors
    deliveredCount = numel(measurements{s, t});
    droppedCount = commStats.droppedByBandwidth(s, t) + ...
        commStats.droppedByLink(s, t) + ...
        commStats.droppedByOutage(s, t);
    total = deliveredCount + droppedCount;
    if total > 0
        % linkQuality 也不设 score floor。若当前传感器信息全部丢失，
        % 它应当在这一时刻被明显降权；真正的硬可用性由 availabilityMask 控制。
        linkQuality(s) = deliveredCount / total;
    end
end
end

function mask = resolveAvailabilityMask(model, commStats, t, numSensors)
mask = ones(1, numSensors);
if nargin >= 2 && isstruct(commStats)
    if isfield(commStats, 'fusionMask') && size(commStats.fusionMask, 1) == numSensors && size(commStats.fusionMask, 2) >= t
        mask = double(commStats.fusionMask(:, t)' > 0);
        return;
    end
    if isfield(commStats, 'activeMask') && size(commStats.activeMask, 1) == numSensors && size(commStats.activeMask, 2) >= t
        mask = double(commStats.activeMask(:, t)' > 0);
        return;
    end
end
if isfield(model, 'adaptiveFusion') && isstruct(model.adaptiveFusion)
    if isfield(model.adaptiveFusion, 'staticMask') && numel(model.adaptiveFusion.staticMask) == numSensors
        mask = double(reshape(model.adaptiveFusion.staticMask, 1, []) > 0);
    end
end
end

function expectedCardinality = computeExpectedCardinality(measurementUpdatedDistributions)
numSensors = numel(measurementUpdatedDistributions);
expectedCardinality = zeros(1, numSensors);
for s = 1:numSensors
    objects = measurementUpdatedDistributions{s};
    if isempty(objects)
        continue;
    end
    expectedCardinality(s) = sum([objects.r]);
end
end

function weights = normalizeScores(score, mask)
maskedScore = score .* mask;
if any(mask > 0) && sum(maskedScore) > 0
    weights = maskedScore / sum(maskedScore);
elseif any(mask > 0)
    weights = mask / sum(mask);
else
    weights = ones(size(score)) / numel(score);
end
end

function weights = enforceMinimumWeight(weights, mask, minWeight)
active = mask > 0;
if ~any(active)
    return;
end
% weight floor 只保护 active 邻居。主实验 8 个传感器且 floor=0.05，
% active floor budget 最多 0.4，仍给动态分数留下足够自由度；如果未来
% 邻居数显著增加，应重新检查 activeCount * minWeight 是否过大。
weights(~active) = 0;
weights(active) = max(weights(active), minWeight);
weights = weights / sum(weights);
end

function value = getField(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function [nu, T] = mprojection(n, measurementUpdatedDistribution)
% 计算 m-projection 后的均值。
nu = zeros(n, 1);
for j = 1:measurementUpdatedDistribution.numberOfGmComponents
    nu = nu + measurementUpdatedDistribution.w(j) * measurementUpdatedDistribution.mu{j};
end
% 计算 m-projection 后的协方差。
T = zeros(n, n);
for j = 1:measurementUpdatedDistribution.numberOfGmComponents
    w = measurementUpdatedDistribution.w(j);
    mu = measurementUpdatedDistribution.mu{j} - nu;
    Sigma = measurementUpdatedDistribution.Sigma{j};
    T = T + w * (Sigma + mu * mu');
end
end
