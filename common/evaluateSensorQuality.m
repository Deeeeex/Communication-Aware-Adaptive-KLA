function [detectionProbability, measurementCovariance, info] = evaluateSensorQuality(model, sensorIdx, targetState, currentTime)
% EVALUATESENSORQUALITY - Resolve per-sensor detection and covariance.
%   Optional model.sensorQuality fields make p_D and Q depend on range and
%   off-axis angle. With sensorQuality disabled, this returns the existing
%   fixed per-sensor values, while still honoring the configured FOV gate.
%   文件导读：
%       几何诱导传感质量变化的集中实现。真值观测生成和传感器关联矩阵
%       都调用这里，因此 p_D、测量协方差、FOV gate 和诊断信息能保持
%       一致。

%% 1. 解析当前传感器位置、速度和目标相对几何
if nargin < 4 || isempty(currentTime)
    currentTime = 1;
end

basePd = resolveSensorValue(model.detectionProbability, sensorIdx, 1.0);
measurementCovariance = model.Q{sensorIdx};

sensorPos = zeros(2, 1);
sensorVel = zeros(2, 1);
if isfield(model, 'sensorTrajectories') && numel(model.sensorTrajectories) >= sensorIdx && ...
        ~isempty(model.sensorTrajectories{sensorIdx})
    traj = model.sensorTrajectories{sensorIdx};
    t = min(max(1, currentTime), size(traj, 2));
    sensorPos = traj(1:2, t);
    if size(traj, 1) >= 4
        sensorVel = traj(3:4, t);
    end
elseif isfield(model, 'sensorInitialStates') && numel(model.sensorInitialStates) >= sensorIdx && ...
        ~isempty(model.sensorInitialStates{sensorIdx})
    sensorState = model.sensorInitialStates{sensorIdx};
    sensorPos = sensorState(1:2);
    if numel(sensorState) >= 4
        sensorVel = sensorState(3:4);
    end
end

targetPos = targetState(1:2);
rel = targetPos - sensorPos;
range = norm(rel);
offAxisDeg = 0;
speed = norm(sensorVel);
if speed > 1e-9 && range > 1e-9
    cosTheta = (rel' * sensorVel) / (range * speed);
    cosTheta = min(max(cosTheta, -1), 1);
    offAxisDeg = acosd(cosTheta);
end

%% 2. FOV gate：先判断目标是否在当前传感器视场内
halfAngleDeg = resolveSensorValueFromField(model, 'sensorFovHalfAngleDeg', sensorIdx, 180);
maxRange = resolveSensorValueFromField(model, 'sensorFovRange', sensorIdx, inf);
inFov = true;
if isfield(model, 'sensorMotionEnabled') && model.sensorMotionEnabled && ...
        isfield(model, 'sensorFovEnabled') && model.sensorFovEnabled
    inFov = range <= maxRange;
    if inFov && speed > 1e-9 && range > 1e-9
        inFov = offAxisDeg <= halfAngleDeg;
    end
end

cfg = struct();
if isfield(model, 'sensorQuality') && isstruct(model.sensorQuality)
    cfg = model.sensorQuality;
end
enabled = getField(cfg, 'enabled', false);

detectionProbability = basePd;
covarianceScale = 1.0;
angleRatio = 0;
if isfinite(halfAngleDeg) && halfAngleDeg > 0
    angleRatio = min(max(offAxisDeg / halfAngleDeg, 0), 1);
end

%% 3. 可选 state-dependent quality：按距离和偏轴角衰减检测率、放大噪声
if enabled
    referenceRange = getField(cfg, 'referenceRange', 120);
    referenceRange = max(referenceRange, eps);
    rangeRatio = max(range / referenceRange, 0);

    detectionRangeDecay = max(getField(cfg, 'detectionRangeDecay', 0.45), 0);
    detectionRangePower = max(getField(cfg, 'detectionRangePower', 1.5), eps);
    anglePenalty = min(max(getField(cfg, 'edgeDetectionPenalty', 0.35), 0), 1);
    anglePower = max(getField(cfg, 'anglePower', 2.0), eps);
    minDetectionProbability = min(max(getField(cfg, 'minDetectionProbability', 0.05), 0), 1);

    rangeDetectionScore = exp(-detectionRangeDecay * rangeRatio^detectionRangePower);
    angleDetectionScore = 1 - anglePenalty * angleRatio^anglePower;
    detectionProbability = basePd * rangeDetectionScore * angleDetectionScore;
    detectionProbability = min(max(detectionProbability, minDetectionProbability), 1);

    rangeNoiseScale = max(getField(cfg, 'rangeNoiseScale', 1.2), 0);
    angleNoiseScale = max(getField(cfg, 'edgeNoiseScale', 1.5), 0);
    minCovarianceScale = max(getField(cfg, 'minCovarianceScale', 1.0), eps);
    maxCovarianceScale = max(getField(cfg, 'maxCovarianceScale', 8.0), minCovarianceScale);
    covarianceScale = 1 + rangeNoiseScale * rangeRatio^detectionRangePower + ...
        angleNoiseScale * angleRatio^anglePower;
    covarianceScale = min(max(covarianceScale, minCovarianceScale), maxCovarianceScale);
    measurementCovariance = covarianceScale * measurementCovariance;
end

%% 4. 视场外目标强制视为不可检测，并返回诊断信息
if ~inFov
    detectionProbability = 0;
end

info = struct();
info.baseDetectionProbability = basePd;
info.range = range;
info.offAxisDeg = offAxisDeg;
info.angleRatio = angleRatio;
info.inFov = inFov;
info.covarianceScale = covarianceScale;
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

function value = resolveSensorValueFromField(model, fieldName, sensorIdx, defaultValue)
if isfield(model, fieldName)
    value = resolveSensorValue(model.(fieldName), sensorIdx, defaultValue);
else
    value = defaultValue;
end
end

function value = getField(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end
