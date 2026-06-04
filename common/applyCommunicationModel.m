function [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig)
% APPLYCOMMUNICATIONMODEL - Apply multi-level communication constraints to measurements.
%   [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig)
%
%   Applies global measurement budget, link loss, and node outage models
%   before measurements are passed to a filter. This function is designed
%   to be called by entry scripts (e.g., runMultisensorFilters).
%   文件导读：
%       观测进入滤波器前的通信约束层。它不会改变 measurements 的整体
%       cell 形状，只在其中删除未送达的测量，并把带宽丢弃、链路丢包、
%       节点 outage、可用性 mask 和 link-quality 统计写入 commStats。
%
%   Inputs
%       measurements - cell array. For multi-sensor: sensors x time cell matrix.
%       model - struct. Used for numberOfSensors if provided.
%       commConfig - struct. Communication configuration (see docs).
%
%   Outputs
%       measurementsDelivered - cell array. Same shape as input measurements.
%       commStats - struct. Basic statistics about drops and outages.

%% 1. 输入兜底和 measurements 形状规范化
if nargin < 3
    commConfig = struct();
end
if nargin < 2
    model = struct();
end

originalSize = size(measurements);
isVector = isvector(measurements);
if isVector
    measurements = reshape(measurements, 1, []);
end

if ~iscell(measurements)
    error('measurements must be a cell array.');
end

numSensors = size(measurements, 1);
numSteps = size(measurements, 2);
if isfield(model, 'numberOfSensors')
    numSensors = model.numberOfSensors;
end

%% 2. 读取通信模型配置和默认值
level = getField(commConfig, 'level', 0);

% Defaults
globalMaxMeasurements = getField(commConfig, 'globalMaxMeasurementsPerStep', []);
if isempty(globalMaxMeasurements)
    globalMaxMeasurements = getField(commConfig, 'globalMaxSensorPacketsPerStep', []);
end
if isempty(globalMaxMeasurements)
    globalMaxMeasurements = inf;
end
sensorWeights = getField(commConfig, 'sensorWeights', ones(1, numSensors));
priorityPolicy = getField(commConfig, 'priorityPolicy', 'weightedPriority');
measurementSelectionPolicy = getField(commConfig, 'measurementSelectionPolicy', 'firstK');

linkModel = getField(commConfig, 'linkModel', 'fixed');
pDrop = getField(commConfig, 'pDrop', 0);
pDropBySensor = getField(commConfig, 'pDropBySensor', []);
pDropLevels = getField(commConfig, 'pDropLevels', []);
pDropLevelCounts = getField(commConfig, 'pDropLevelCounts', []);
pGoodToBad = getField(commConfig, 'pGoodToBad', 0.1);
pBadToGood = getField(commConfig, 'pBadToGood', 0.3);
pDropGood = getField(commConfig, 'pDropGood', 0.05);
pDropBad = getField(commConfig, 'pDropBad', 0.4);

maxOutageNodes = getField(commConfig, 'maxOutageNodes', 1);
outageSchedule = getField(commConfig, 'outageSchedule', []);
outageMinDuration = getField(commConfig, 'outageMinDuration', 10);
outageMaxDuration = getField(commConfig, 'outageMaxDuration', 30);

%% 3. 规范化 per-sensor 权重和丢包率配置
if numel(sensorWeights) ~= numSensors
    sensorWeights = ones(1, numSensors);
end
if isempty(pDropBySensor)
    pDropBySensor = resolveTieredPDrop(pDropLevels, pDropLevelCounts, numSensors);
end
if ~isempty(pDropBySensor)
    pDropBySensor = reshape(pDropBySensor, 1, []);
    if numel(pDropBySensor) ~= numSensors
        pDropBySensor = [];
    else
        pDropBySensor = min(max(pDropBySensor, 0), 1);
    end
end

%% 4. 初始化 delivered measurements 和统计字段
measurementsDelivered = measurements;

commStats = struct();
commStats.allowedPacketsPerStep = zeros(1, numSteps);
commStats.allowedMeasurementsPerStep = zeros(1, numSteps);
commStats.droppedByBandwidth = zeros(numSensors, numSteps);
commStats.droppedByLink = zeros(numSensors, numSteps);
commStats.droppedByOutage = zeros(numSensors, numSteps);
commStats.linkState = zeros(numSensors, numSteps); % 0=good, 1=bad
commStats.outageSchedule = outageSchedule;
if ~isempty(pDropBySensor)
    commStats.pDropBySensor = pDropBySensor;
else
    commStats.pDropBySensor = pDrop * ones(1, numSensors);
end

%% 5. Level 0 表示理想通信：直接返回原始观测
if level <= 0
    if isVector
        measurementsDelivered = reshape(measurementsDelivered, originalSize);
    end
    return;
end

if level >= 3 && isempty(outageSchedule)
    outageSchedule = generateRandomOutage(numSensors, numSteps, maxOutageNodes, outageMinDuration, outageMaxDuration);
    commStats.outageSchedule = outageSchedule;
end

if strcmpi(linkModel, 'markov')
    isBad = false(numSensors, 1);
end

for t = 1:numSteps
    % Clear delivered packets for this time step
    for s = 1:numSensors
        measurementsDelivered{s, t} = {};
    end

    % Identify sensors with non-empty packets
    hasPacket = false(numSensors, 1);
    for s = 1:numSensors
        hasPacket(s) = numel(measurements{s, t}) > 0;
    end

    % Level 1: global measurement budget
    remainingBudget = globalMaxMeasurements;
    allowedSensors = false(numSensors, 1);
    if level >= 1
        candidates = find(hasPacket);
        ordered = prioritizeSensors(candidates, sensorWeights, priorityPolicy);
        for idx = 1:numel(ordered)
            s = ordered(idx);
            if remainingBudget <= 0
                break;
            end
            count = numel(measurements{s, t});
            if count <= remainingBudget
                measurementsDelivered{s, t} = measurements{s, t};
                remainingBudget = remainingBudget - count;
                allowedSensors(s) = true;
            else
                measurementsDelivered{s, t} = selectMeasurements(measurements{s, t}, remainingBudget, measurementSelectionPolicy);
                commStats.droppedByBandwidth(s, t) = count - remainingBudget;
                remainingBudget = 0;
                allowedSensors(s) = true;
            end
        end
        if remainingBudget <= 0
            droppedSensors = hasPacket & ~allowedSensors;
            for s = find(droppedSensors)'
                commStats.droppedByBandwidth(s, t) = numel(measurements{s, t});
            end
        end
    else
        for s = find(hasPacket)'
            measurementsDelivered{s, t} = measurements{s, t};
        end
        allowedSensors = hasPacket;
    end

    commStats.allowedPacketsPerStep(t) = sum(allowedSensors);
    commStats.allowedMeasurementsPerStep(t) = sum(cellfun(@numel, measurementsDelivered(:, t)));

    % Level 2: link loss
    if level >= 2
        for s = find(allowedSensors)'
            deliveredCount = numel(measurementsDelivered{s, t});
            if deliveredCount == 0
                continue;
            end
            dropPacket = false;
            if strcmpi(linkModel, 'markov')
                if isBad(s)
                    if rand < pBadToGood
                        isBad(s) = false;
                    end
                else
                    if rand < pGoodToBad
                        isBad(s) = true;
                    end
                end
                commStats.linkState(s, t) = isBad(s);
                if isBad(s)
                    dropPacket = rand < pDropBad;
                else
                    dropPacket = rand < pDropGood;
                end
            else
                if ~isempty(pDropBySensor)
                    dropPacket = rand < pDropBySensor(s);
                else
                    dropPacket = rand < pDrop;
                end
            end
            if dropPacket
                commStats.droppedByLink(s, t) = deliveredCount;
                measurementsDelivered{s, t} = {};
            end
        end
    end

    % Level 3: node outage (max nodes)
    if level >= 3 && ~isempty(outageSchedule)
        outagedSensors = sensorsOutagedAt(outageSchedule, t);
        for s = outagedSensors(:)'
            deliveredCount = numel(measurementsDelivered{s, t});
            commStats.droppedByOutage(s, t) = deliveredCount;
            measurementsDelivered{s, t} = {};
        end
    end
end

if isVector
    measurementsDelivered = reshape(measurementsDelivered, originalSize);
end
end

function pDropBySensor = resolveTieredPDrop(pDropLevels, pDropLevelCounts, numSensors)
pDropBySensor = [];
if isempty(pDropLevels) || isempty(pDropLevelCounts)
    return;
end

pDropLevels = reshape(pDropLevels, 1, []);
pDropLevelCounts = round(reshape(pDropLevelCounts, 1, []));
if numel(pDropLevels) ~= numel(pDropLevelCounts)
    return;
end
if any(pDropLevelCounts < 0) || sum(pDropLevelCounts) ~= numSensors
    return;
end

pDropLevels = min(max(pDropLevels, 0), 1);
assigned = zeros(1, numSensors);
cursor = 1;
for i = 1:numel(pDropLevels)
    count = pDropLevelCounts(i);
    if count <= 0
        continue;
    end
    assigned(cursor:cursor + count - 1) = pDropLevels(i);
    cursor = cursor + count;
end

if cursor - 1 ~= numSensors
    return;
end

pDropBySensor = assigned(randperm(numSensors));
end

function value = getField(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function ordered = prioritizeSensors(sensors, weights, policy)
sensorWeights = weights(sensors);
switch lower(policy)
    case 'weightedshuffle'
        % Sort by weight, shuffle within equal-weight groups
        [sortedWeights, sortIdx] = sort(sensorWeights, 'descend');
        ordered = sensors(sortIdx);
        uniqueWeights = unique(sortedWeights);
        for i = 1:numel(uniqueWeights)
            w = uniqueWeights(i);
            idx = find(sortedWeights == w);
            if numel(idx) > 1
                shuffled = idx(randperm(numel(idx)));
                ordered(idx) = ordered(shuffled);
            end
        end
    otherwise
        % weightedPriority
        [~, sortIdx] = sort(sensorWeights, 'descend');
        ordered = sensors(sortIdx);
end
end

function selected = selectMeasurements(measurementsCell, k, policy)
if k <= 0
    selected = {};
    return;
end
count = numel(measurementsCell);
if k >= count
    selected = measurementsCell;
    return;
end
switch lower(policy)
    case 'random'
        idx = randperm(count, k);
        selected = measurementsCell(idx);
    otherwise
        % firstK
        selected = measurementsCell(1:k);
end
end

function schedule = generateRandomOutage(numSensors, numSteps, maxOutageNodes, minDuration, maxDuration)
schedule = struct('sensor', {}, 'start', {}, 'end', {});
if maxOutageNodes <= 0 || numSensors <= 0
    return;
end

numOutageNodes = min(maxOutageNodes, numSensors);
sensorIds = randperm(numSensors, numOutageNodes);
for i = 1:numOutageNodes
    duration = randi([minDuration, maxDuration]);
    if duration >= numSteps
        startTime = 1;
        endTime = numSteps;
    else
        startTime = randi([1, numSteps - duration + 1]);
        endTime = startTime + duration - 1;
    end
    schedule(end+1).sensor = sensorIds(i);
    schedule(end).start = startTime;
    schedule(end).end = endTime;
end
end

function sensors = sensorsOutagedAt(schedule, t)
sensors = [];
for i = 1:numel(schedule)
    if t >= schedule(i).start && t <= schedule(i).end
        sensors(end+1) = schedule(i).sensor; %#ok<AGROW>
    end
end
end
