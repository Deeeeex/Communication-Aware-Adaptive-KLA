function [measurementsScheduled, samplingStats] = applyMultiRateSensorSchedule(measurements, samplingPeriods, phaseOffsets)
% APPLYMULTIRATESENSORSCHEDULE - Gate measurements by per-sensor sampling period.
%   Non-sampling instants are marked separately from scheduled scans with no
%   detections so multi-rate experiments can propagate stale posteriors
%   without treating every skipped scan as a missed detection.
%   文件导读：
%       多速率传感器的预调度器。未采样时刻会被标记为 prediction-only，
%       这样滤波器不会把“没采样”误当成“采样但没有检测”。返回的 age
%       和 mask 诊断可供 freshness / information-decay 权重使用。

%% 1. 没有调度配置时直接透传 measurements
if nargin < 2 || isempty(samplingPeriods)
    measurementsScheduled = measurements;
    samplingStats = struct();
    return;
end

%% 2. 规范化采样周期和相位偏移
if ~iscell(measurements)
    error('measurements must be a cell array.');
end

numSensors = size(measurements, 1);
numSteps = size(measurements, 2);
samplingPeriods = round(reshape(samplingPeriods, 1, []));
if numel(samplingPeriods) ~= numSensors
    error('samplingPeriods must have one entry per sensor.');
end
samplingPeriods = max(samplingPeriods, 1);

if nargin < 3 || isempty(phaseOffsets)
    phaseOffsets = zeros(1, numSensors);
end
phaseOffsets = round(reshape(phaseOffsets, 1, []));
if numel(phaseOffsets) ~= numSensors
    error('phaseOffsets must have one entry per sensor.');
end
phaseOffsets = mod(phaseOffsets, samplingPeriods);

%% 3. 按传感器逐时刻应用采样周期；未采样时刻清空观测并记录 age
measurementsScheduled = measurements;
sensorSampleMask = false(numSensors, numSteps);
sensorSampleAge = zeros(numSensors, numSteps);
droppedBySchedule = zeros(numSensors, numSteps);

for s = 1:numSensors
    lastSampleStep = 1;
    for t = 1:numSteps
        isSampleStep = mod((t - 1) - phaseOffsets(s), samplingPeriods(s)) == 0;
        sensorSampleMask(s, t) = isSampleStep;
        if isSampleStep
            lastSampleStep = t;
            sensorSampleAge(s, t) = 0;
        else
            droppedBySchedule(s, t) = numel(measurements{s, t});
            measurementsScheduled{s, t} = {};
            sensorSampleAge(s, t) = t - lastSampleStep;
        end
    end
end

samplingStats = struct();
samplingStats.sensorSampleMask = sensorSampleMask;
samplingStats.sensorSampleAge = sensorSampleAge;
samplingStats.droppedBySchedule = droppedBySchedule;
samplingStats.samplingPeriods = samplingPeriods;
samplingStats.samplingPhaseOffsets = phaseOffsets;
end
