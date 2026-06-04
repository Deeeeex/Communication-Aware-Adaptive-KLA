function [stateEstimatesBySensor, localModels, neighborMap] = runDistributedLmbFilter(model, measurements, sensorTrajectories, neighborMap, commStats)
% RUNDISTRIBUTEDLMBFILTER - Run local fusion at each sensor using neighbor data
%   [stateEstimatesBySensor, localModels, neighborMap] =
%   runDistributedLmbFilter(model, measurements, sensorTrajectories, neighborMap)
%
%   Each sensor performs local fusion using its own measurements plus those
%   received from neighboring sensors. There is no centralized fusion.
%   文件导读：
%       分布式本地融合的包装层。它不直接计算动态权重，而是为每个传感器
%       构造邻域 sub-model、初始化 Metropolis/Uniform 拓扑权重、生成
%       spatial/existence 结构先验，再调用 runParallelUpdateLmbFilter 在本地
%       邻域内执行 GA/AA/PU 融合。
%       “邻域 sub-model”不是新的物理模型，而是从全局 multi-sensor model
%       切出来的局部多传感器问题：只包含当前 sensor 能通信到的 neighborIdx
%       对应的 C/Q/p_D/clutter、测量、通信统计和融合权重。每个全局 sensor
%       都独立拥有一份这样的 local model，因此输出也是 per-sensor estimate。
%
%   主调用链：
%       neighborMap / weightsBySensor
%       -> buildSubModel(model, neighborIdx)
%       -> slice measurements / trajectories / commStats
%       -> runParallelUpdateLmbFilter(localModel, localMeasurements, ...)
%       -> stateEstimatesBySensor{s}
%
%   Inputs
%       model - full multi-sensor model
%       measurements - cell array [S x T] of measurements
%       sensorTrajectories - cell array of sensor trajectories (optional)
%       neighborMap - cell array of neighbor indices per sensor (optional)
%
%   Outputs
%       stateEstimatesBySensor - cell array of state estimates per sensor
%       localModels - cell array of per-sensor sub-models
%       neighborMap - resolved neighbor map

numberOfSensors = model.numberOfSensors;
if nargin < 3
    sensorTrajectories = [];
end
%% 1. 确定每个传感器的通信邻域
% neighborMap{s} 存的是“全局 sensor s 可以融合哪些全局 sensor 的信息”。
% 如果实验脚本显式传入 4+4 邻接关系，就直接使用；否则根据通信半径
% 和传感器初始位置动态计算。
if nargin < 4 || isempty(neighborMap)
    neighborMap = computeNeighborMap(model, sensorTrajectories);
end

stateEstimatesBySensor = cell(1, numberOfSensors);
localModels = cell(1, numberOfSensors);

%% 2. 根据邻域图初始化固定拓扑权重：Metropolis 或 Uniform
% 这些是每个 local model 的初始融合权重，也是 adaptiveFusion 关闭时
% 的固定权重 baseline。adaptiveFusion 打开后，它们仍作为 EMA 初值、
% 拓扑 prior 和 fallback 权重使用。
if isfield(model, 'fusionWeighting')
    fusionWeighting = model.fusionWeighting;
else
    fusionWeighting = 'Uniform';
end
if strcmpi(fusionWeighting, 'Metropolis')
    weightsBySensor = computeMetropolisWeights(neighborMap);
else
    weightsBySensor = computeUniformWeights(neighborMap);
end

%% 3. 对每个传感器构造局部问题，并在其邻域内独立运行并行更新滤波器
for s = 1:numberOfSensors
    neighborIdx = neighborMap{s};
    % 3.1 sub-model 只保留该传感器可见邻域内的传感器参数。
    % 例如全局 sensor 1 的 neighborIdx=[1 2 3 4 5] 时，localModels{1}
    % 就是一个 5-sensor LMB 问题；其中 local sensor index 1..5 分别
    % 对应全局 sensor 1,2,3,4,5。
    localModels{s} = buildSubModel(model, neighborIdx);
    % 3.2 初始化 GA/AA scalar、spatial、existence 权重。
    % runParallelUpdateLmbFilter 会把这些字段读入 prevWeights；如果当前
    % arm 不启用动态权重，则 merging 端直接消费这些固定拓扑权重。
    localModels{s}.gaSensorWeights = weightsBySensor{s};
    localModels{s}.aaSensorWeights = weightsBySensor{s};
    localModels{s}.gaSpatialWeights = weightsBySensor{s};
    localModels{s}.aaSpatialWeights = weightsBySensor{s};
    localModels{s}.gaExistenceWeights = weightsBySensor{s};
    localModels{s}.aaExistenceWeights = weightsBySensor{s};
    % 3.3 结构先验是动态权重 structure-aware 分支的输入，不直接替代质量因子。
    % spatial prior 偏向邻域结构相似/冗余的节点；existence prior 只做
    % 轻量 novelty 调制。二者都会归一到均值为 1，避免整体放大 score。
    [spatialStructurePrior, existenceStructurePrior] = computeLocalStructurePriors(neighborMap, s, neighborIdx);
    localModels{s}.gaTopologyWeights = weightsBySensor{s};
    localModels{s}.aaTopologyWeights = weightsBySensor{s};
    localModels{s}.gaSpatialStructurePrior = spatialStructurePrior;
    localModels{s}.aaSpatialStructurePrior = spatialStructurePrior;
    localModels{s}.gaExistenceStructurePrior = existenceStructurePrior;
    localModels{s}.aaExistenceStructurePrior = existenceStructurePrior;
    % 3.4 measurements 和 commStats 都要切片到局部邻域，保持索引和 sub-model 对齐。
    % 后续 computeAdaptiveFusionWeights 看到的第 k 个传感器，是局部模型
    % 的第 k 个传感器，不再是全局 sensor k。
    localMeasurements = measurements(neighborIdx, :);
    localSensorTraj = [];
    if ~isempty(sensorTrajectories)
        localSensorTraj = sensorTrajectories(neighborIdx);
        localModels{s}.sensorTrajectories = localSensorTraj;
    end
    if isfield(localModels{s}, 'adaptiveFusion') && isstruct(localModels{s}.adaptiveFusion)
        % progressLabel 只用于长实验进度打印，帮助定位当前 local filter。
        localModels{s}.adaptiveFusion.progressLabel = sprintf('sensor %d', s);
    end
    localCommStats = [];
    if nargin >= 5 && isstruct(commStats)
        localCommStats = sliceCommStats(commStats, neighborIdx);
    end
    % 3.5 真正运行局部多传感器 LMB。返回值是“全局 sensor s 视角下”的
    % 融合后轨迹估计，而不是邻域内每个传感器各自的估计。
    stateEstimatesBySensor{s} = runParallelUpdateLmbFilter(localModels{s}, localMeasurements, localCommStats, localSensorTraj);
end
end

function [spatialPrior, existencePrior] = computeLocalStructurePriors(neighborMap, sourceSensorIdx, sensorIdx)
% 中文导读：
%   根据邻域重叠度生成两个结构先验。spatial 分支偏向结构冗余的邻居，
%   因为冗余邻居更容易提供稳定空间一致性；existence 分支只给轻量
%   novelty 偏好，避免拓扑修正过度影响基数判断。
nLocal = numel(sensorIdx);
spatialPrior = ones(1, nLocal);
existencePrior = ones(1, nLocal);
if nargin < 2 || isempty(neighborMap) || isempty(sensorIdx) || sourceSensorIdx > numel(neighborMap)
    return;
end

localSet = reshape(sensorIdx, 1, []);
sourceNeighborhood = unique(neighborMap{sourceSensorIdx});
for k = 1:nLocal
    globalSensorIdx = localSet(k);
    if globalSensorIdx > numel(neighborMap) || isempty(neighborMap{globalSensorIdx})
        continue;
    end
    targetNeighborhood = unique(neighborMap{globalSensorIdx});
    sharedNeighbors = numel(intersect(sourceNeighborhood, targetNeighborhood));
    totalNeighbors = numel(union(sourceNeighborhood, targetNeighborhood));
    if totalNeighbors <= 0
        similarity = 0;
    else
        similarity = sharedNeighbors / totalNeighbors;
    end

    % similarity 越高，说明该 neighbor 和 source sensor 的邻域重叠越多。
    % spatial fusion 偏好这种结构冗余，因为它通常意味着空间估计更稳定；
    % existence fusion 则给“不完全相同视角”的 neighbor 一点 novelty 偏好，
    % 但只用 0.5 系数，避免拓扑项过度影响 cardinality。
    spatialPrior(k) = 1 + similarity;
    existencePrior(k) = 1 + 0.5 * (1 - similarity);
end

spatialPrior = spatialPrior / mean(spatialPrior);
existencePrior = existencePrior / mean(existencePrior);
end

function neighborMap = computeNeighborMap(model, sensorTrajectories)
% 中文导读：
%   根据传感器初始位置和通信半径生成邻域图。没有可达邻居时至少保留
%   自身，保证每个 local filter 都有合法传感器集合。
    numberOfSensors = model.numberOfSensors;
    neighborMap = cell(1, numberOfSensors);
    if isfield(model, 'sensorCommRange')
        commRange = model.sensorCommRange;
    else
        commRange = 50;
    end

    positions = zeros(2, numberOfSensors);
    if ~isempty(sensorTrajectories)
        for s = 1:numberOfSensors
            positions(:, s) = sensorTrajectories{s}(1:2, 1);
        end
    elseif isfield(model, 'sensorInitialStates') && ~isempty(model.sensorInitialStates)
        for s = 1:numberOfSensors
            positions(:, s) = model.sensorInitialStates{s}(1:2);
        end
    else
        positions = zeros(2, numberOfSensors);
    end

    for s = 1:numberOfSensors
        deltas = positions - positions(:, s);
        dists = sqrt(sum(deltas.^2, 1));
        neighborIdx = find(dists <= commRange);
        if isempty(neighborIdx)
            neighborIdx = s;
        end
        neighborMap{s} = neighborIdx;
    end
end

function subModel = buildSubModel(model, sensorIdx)
% 中文导读：
%   从全局 model 中切出局部邻域 model。动态权重和融合函数都按局部
%   sensor index 工作，因此 C/Q/clutter/p_D 必须同步切片。
%   注意：object birth、motion model、OSPA 参数等和传感器无关的字段仍
%   保留全局值；只有传感器维度的字段需要按 sensorIdx 裁剪。
    subModel = model;
    subModel.numberOfSensors = numel(sensorIdx);
    subModel.C = model.C(sensorIdx);
    subModel.Q = model.Q(sensorIdx);
    subModel.clutterRate = model.clutterRate(sensorIdx);
    subModel.clutterPerUnitVolume = model.clutterPerUnitVolume(sensorIdx);
    subModel.detectionProbability = model.detectionProbability(sensorIdx);
end

function localCommStats = sliceCommStats(commStats, sensorIdx)
% 中文导读：
%   commStats 的矩阵字段也要按邻域切片，否则动态权重会把局部传感器
%   权重和全局通信统计错位。
%   比如 local model 的第 2 个 sensor 可能是全局 sensor 6；切片后，
%   localCommStats.droppedByLink(2,:) 才能和 localMeasurements(2,:) 对齐。
    localCommStats = commStats;
    if isfield(commStats, 'pDropBySensor') && numel(commStats.pDropBySensor) >= max(sensorIdx)
        localCommStats.pDropBySensor = commStats.pDropBySensor(sensorIdx);
    end
    if isfield(commStats, 'droppedByBandwidth')
        localCommStats.droppedByBandwidth = commStats.droppedByBandwidth(sensorIdx, :);
    end
    if isfield(commStats, 'droppedByLink')
        localCommStats.droppedByLink = commStats.droppedByLink(sensorIdx, :);
    end
    if isfield(commStats, 'droppedByOutage')
        localCommStats.droppedByOutage = commStats.droppedByOutage(sensorIdx, :);
    end
    if isfield(commStats, 'innovationConsistency')
        localCommStats.innovationConsistency = commStats.innovationConsistency(sensorIdx, :);
    end
    if isfield(commStats, 'associationAmbiguityScore')
        localCommStats.associationAmbiguityScore = commStats.associationAmbiguityScore(sensorIdx, :);
    end
    if isfield(commStats, 'sensorSampleMask')
        localCommStats.sensorSampleMask = commStats.sensorSampleMask(sensorIdx, :);
    end
    if isfield(commStats, 'sensorSampleAge')
        localCommStats.sensorSampleAge = commStats.sensorSampleAge(sensorIdx, :);
    end
    if isfield(commStats, 'droppedBySchedule')
        localCommStats.droppedBySchedule = commStats.droppedBySchedule(sensorIdx, :);
    end
    if isfield(commStats, 'samplingPeriods') && numel(commStats.samplingPeriods) >= max(sensorIdx)
        localCommStats.samplingPeriods = commStats.samplingPeriods(sensorIdx);
    end
    if isfield(commStats, 'samplingPhaseOffsets') && numel(commStats.samplingPhaseOffsets) >= max(sensorIdx)
        localCommStats.samplingPhaseOffsets = commStats.samplingPhaseOffsets(sensorIdx);
    end
end

function weightsBySensor = computeUniformWeights(neighborMap)
% COMPUTEUNIFORMWEIGHTS - 每个邻居等权。
% 这是最简单的固定融合权重，用于非 Metropolis baseline 或兜底。
    numberOfSensors = numel(neighborMap);
    weightsBySensor = cell(1, numberOfSensors);
    for s = 1:numberOfSensors
        n = numel(neighborMap{s});
        weightsBySensor{s} = ones(1, n) / n;
    end
end

function weightsBySensor = computeMetropolisWeights(neighborMap)
% COMPUTEMETROPOLISWEIGHTS - 根据邻域度数构造 row-stochastic Metropolis 权重。
% 对 sensor s 的每个邻居 j，非自身权重为 1/(1+max(deg(s),deg(j)))；
% 自身权重补足到总和 1。这样每个 local filter 都得到一组合法 convex weights。
    numberOfSensors = numel(neighborMap);
    degrees = zeros(1, numberOfSensors);
    for s = 1:numberOfSensors
        degrees(s) = numel(neighborMap{s});
    end
    weightsBySensor = cell(1, numberOfSensors);
    for s = 1:numberOfSensors
        neighborIdx = neighborMap{s};
        nLocal = numel(neighborIdx);
        w = zeros(1, nLocal);
        sumOff = 0;
        for k = 1:nLocal
            j = neighborIdx(k);
            if j ~= s
                w(k) = 1 / (1 + max(degrees(s), degrees(j)));
                sumOff = sumOff + w(k);
            end
        end
        selfIdx = find(neighborIdx == s, 1, 'first');
        if isempty(selfIdx)
            neighborIdx(end+1) = s;
            w(end+1) = 0;
            selfIdx = numel(w);
        end
        w(selfIdx) = max(0, 1 - sumOff);
        if abs(sum(w) - 1) > 1e-6
            w = w / sum(w);
        end
        weightsBySensor{s} = w;
    end
end
