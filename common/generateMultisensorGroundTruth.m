function [groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model, varargin)
% GENERATEMULTISENSORGROUNDTRUTH -- Generates measurements and groundtruth for a simple hard-coded example.
%   [groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model)
%   [groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model)
%
%   Generates the objects' groundtruths for a simple scenario, and also their measurements.
%   Supports mobile sensors with configurable motion models.
%   文件导读：
%       多传感器场景生成器。它负责生成目标真值、传感器轨迹、每个传感器
%       的检测/漏检/杂波观测，以及评估用 RFS truth。移动传感器、FOV
%       和 state-dependent sensing variation 都会在这里影响观测流。
%
%   See also generateMultisensorModel, plotMultisensorResults
%
%   Inputs
%       model - struct. A struct with the fields declared in generateMultisensorModel.
%       numberOfObjects - integer. The number of objects to be simulated
%           for a 'Random' scenario.
%
%   Output
%       groundTruth - cell array. An array of each object's groundtruth
%           trajectory.
%       measurements - cell array. An array containing the measurements for
%           each time-step of the simulation.
%       groundTruthRfs - struct. Groundtruth RFS and optimal Kalman filter
%           output in RFS form.
%       sensorTrajectories - cell array. Each sensor's trajectory over time
%           (only if sensorMotionEnabled).

simulationLength = 100;
if isfield(model, 'simulationLength') && ~isempty(model.simulationLength)
    simulationLength = max(1, round(model.simulationLength));
end


%% 1. 定义目标真值：固定场景、随机场景或 formation 场景
if (strcmp(model.scenarioType, 'Fixed'))
    numberOfObjects = 10;
    % Object birth times
    objectBirthTimes = [1, 1, 20, 20, 40, 40, 60, 60, 60, 60];
    objectDeathTimes = [70, 70, 80, 80, 90, 90, 100, 100, 100, 100];
    % Object birth states
    birthLocationIndex = [1 2 3 4 1 4 1 2 3 4];
    priorLocations = [-80.0 -20.0 0.75 1.5;
        -20.0 80.0 -1.0 -2.0;
        0.0 0.0 -0.5 -1.0;
        40.0 -60.0 -0.25 -0.5;
        -80.0 -20.0 1.0 1.0;
        40.0 -60.0 -1.0 2.0;
        -80.0 -20.0 1.0 -0.5;
        -20.0 80.0 1.0 -1.0;
        0.0 0.0 1.0 -1.0;
        40.0 -60.0 -1.0 0.5]';
elseif (strcmp(model.scenarioType, 'Random'))
    % Number of objects
    if (nargin == 2)
        numberOfObjects = varargin{1};
    else
        error('You must specify the number of objects for a Random scenario');
    end
     % Object birth times
    objectBirthTimes = ones(1, numberOfObjects); %randi([1 simulationLength], 1, numberOfObjects);
    objectDeathTimes = simulationLength * ones(1, numberOfObjects); %randi([1 simulationLength], 1, numberOfObjects) + objectBirthTimes + model.minimumTrajectoryLength;
    objectDeathTimes(objectDeathTimes > simulationLength) = simulationLength;
    % Object birth states
    birthLocationIndex = 1:model.numberOfBirthLocations; %randi([1 model.numberOfBirthLocations],1 , numberOfObjects);
    priorLocations = [model.muB{birthLocationIndex}];
    priorLocations(3:4, :) = 3 * randn(numberOfObjects, 2)';
elseif (strcmp(model.scenarioType, 'Formation'))
    % Formation targets (right side)
    if isfield(model, 'targetFormationCount')
        numberOfObjects = model.targetFormationCount;
    else
        numberOfObjects = model.numberOfBirthLocations;
    end
    if ~isfield(model, 'targetFormationStartTime')
        model.targetFormationStartTime = 1;
    end
    if ~isfield(model, 'targetFormationBirthInterval')
        model.targetFormationBirthInterval = 5;
    end
    if ~isfield(model, 'targetFormationStaggeredBirths')
        model.targetFormationStaggeredBirths = false;
    end
    if ~isfield(model, 'targetFormationLifeSpan')
        model.targetFormationLifeSpan = simulationLength;
    end
    birthLocationIndex = 1:numberOfObjects;
    priorLocations = [model.muB{birthLocationIndex}];
    if model.targetFormationStaggeredBirths
        objectBirthTimes = model.targetFormationStartTime + (0:numberOfObjects-1) * model.targetFormationBirthInterval;
    else
        objectBirthTimes = model.targetFormationStartTime * ones(1, numberOfObjects);
    end
    objectBirthTimes(objectBirthTimes < 1) = 1;
    objectBirthTimes(objectBirthTimes > simulationLength) = simulationLength;
    objectDeathTimes = min(objectBirthTimes + model.targetFormationLifeSpan - 1, simulationLength);
end
%% 2. 分配输出容器：每个传感器、每个时刻一个 measurement cell
measurements = repmat({}, model.numberOfSensors, simulationLength);
groundTruth = repmat({}, numberOfObjects, 1);
groundTruthRfs.x = repmat({{}}, 1, simulationLength);
groundTruthRfs.mu = groundTruthRfs.x;
groundTruthRfs.Sigma = groundTruthRfs.x;
groundTruthRfs.cardinality = zeros(1, simulationLength);
%% 3. 先生成 per-sensor 杂波测量；目标测量随后追加
for i = 1:simulationLength
    for s = 1:model.numberOfSensors
        numberOfClutterMeasurements = poissrnd(model.clutterRate(s));
        measurements{s, i} = cell(numberOfClutterMeasurements, 1);
        for j = 1:numberOfClutterMeasurements
            measurements{s, i}{j} = model.observationSpaceLimits(:, 1) + 2 * model.observationSpaceLimits(:, 2) .* rand(model.zDimension, 1);
        end
    end
end
%% 4. 初始化传感器轨迹：静态传感器返回空轨迹，移动传感器返回完整轨迹容器
if model.sensorMotionEnabled
    sensorTrajectories = cell(1, model.numberOfSensors);
    for s = 1:model.numberOfSensors
        sensorTrajectories{s} = zeros(4, simulationLength);
        if isfield(model, 'sensorInitialStates') && numel(model.sensorInitialStates) >= s ...
                && ~isempty(model.sensorInitialStates{s})
            sensorTrajectories{s}(:, 1) = model.sensorInitialStates{s};
        else
            sensorTrajectories{s}(:, 1) = zeros(4, 1);
        end
    end
else
    sensorTrajectories = [];
end
%% 5. 根据 CV/CT/Formation 运动模型推进传感器轨迹
if model.sensorMotionEnabled
    if isfield(model, 'sensorMotionType')
        motionType = upper(model.sensorMotionType);
    else
        motionType = 'CV';
    end
    switch motionType
        case 'CT'
            sensorTrajectories = generateCtSensorTrajectories(model, simulationLength);
        case 'FORMATION'
            sensorTrajectories = generateFormationSensorTrajectories(model, simulationLength);
        otherwise
            for t = 2:simulationLength
                for s = 1:model.numberOfSensors
                    % CV motion model for sensors
                    sensorTrajectories{s}(:, t) = model.A * sensorTrajectories{s}(:, t-1) + ...
                        sampleSensorProcessNoise(model, s);
                end
            end
    end
end
if model.sensorMotionEnabled
    model.sensorTrajectories = sensorTrajectories;
end

%% 6. 逐目标传播真值，并为每个传感器生成目标测量
for i = 1:numberOfObjects
    % Initialise the object's trajectory
    trajectoryLength = objectDeathTimes(i) - objectBirthTimes(i) + 1;
    t = objectBirthTimes(i);
    x =  priorLocations(:, i);
    mu = model.muB{birthLocationIndex(i)};
    Sigma = model.SigmaB{birthLocationIndex(i)};
    groundTruth{i} = repmat([t; x], 1, trajectoryLength);
    % Simulate the object's trajectory
    for j = 1:trajectoryLength
        % Predict the object's state
        if (j > 1)
            % Point estimate
            x = model.A * x + model.u;
            t = t + 1;
            groundTruth{i}(:, j) = [t; x];
            % Kalman filter prediction
            mu = model.A * mu + model.u;
            Sigma = model.A * Sigma * model.A' + model.R;
        end
        % Preallocate parameters
        generatedMeasurement = false(model.numberOfSensors, 1);
        measurementCovariances = cell(model.numberOfSensors, 1);
        for s = 1:model.numberOfSensors
            [pdSensor, qSensor] = evaluateSensorQuality(model, s, x, t);
            generatedMeasurement(s) = rand < pdSensor;
            measurementCovariances{s} = qSensor;
        end
        numberOfAssignments = sum(generatedMeasurement);
        if (numberOfAssignments)
            counter = 0;
            z = zeros(model.zDimension * numberOfAssignments, 1);
            C = zeros(model.zDimension * numberOfAssignments, model.xDimension);
            % Generate measurements
            for s = 1:model.numberOfSensors
                % 根据当前传感器的 state-dependent 检测概率决定是否生成测量。
                if (generatedMeasurement(s))
                    qSensor = measurementCovariances{s};
                    if model.sensorMotionEnabled
                        % Mobile sensor: measure relative position to sensor
                        sensorPos = sensorTrajectories{s}(1:2, t);
                        targetPos = x(1:2);
                        relativePos = targetPos - sensorPos;
                        y = sensorPos + model.C{s} * [relativePos; 0; 0] + ...
                            chol(qSensor, 'lower') * randn(1, model.zDimension)';
                    else
                        % Static sensor (original code)
                        y = model.C{s} * x + chol(qSensor, 'lower') * randn(1, model.zDimension)';
                    end
                    sensorMeasurements = measurements{s, t};
                    sensorMeasurements{numel(sensorMeasurements) + 1} = y;
                    measurements{s, t} = sensorMeasurements;
                    % Stack measurements and matrices
                    start = model.zDimension * counter + 1;
                    finish = start + model.zDimension - 1;
                    z(start:finish) = y;
                    C(start:finish, :) = model.C{s};
                    counter = counter + 1;
                end
            end
            % Multisensor Kalman filter update
            Q = blkdiag(measurementCovariances{generatedMeasurement});
            K = Sigma * C' / ( C * Sigma * C' + Q);
            mu = mu + K * (z - C * mu);
            Sigma = (eye(model.xDimension) - K * C) * Sigma;
        end
        % Add to RFS
        groundTruthRfs.x{t}{end+1} = x;
        groundTruthRfs.mu{t}{end+1} = mu;
        groundTruthRfs.Sigma{t}{end+1} = Sigma;
        groundTruthRfs.cardinality(t) = groundTruthRfs.cardinality(t) + 1;
    end
end
end

function sensorTrajectories = generateCtSensorTrajectories(model, simulationLength)
% GENERATECTSENSORTRAJECTORIES - Coordinated turn sensor motion
    sensorTrajectories = cell(1, model.numberOfSensors);
    for s = 1:model.numberOfSensors
        sensorTrajectories{s} = zeros(4, simulationLength);
        if isfield(model, 'sensorInitialStates') && numel(model.sensorInitialStates) >= s ...
                && ~isempty(model.sensorInitialStates{s})
            sensorTrajectories{s}(:, 1) = model.sensorInitialStates{s};
        else
            sensorTrajectories{s}(:, 1) = [0; 0; 0; 0];
        end
        omega = 0;
        if isfield(model, 'sensorTurnRate')
            omega = model.sensorTurnRate;
        end
        for t = 2:simulationLength
            if isfield(model, 'sensorTurnModel') && strcmp(model.sensorTurnModel, 'WhiteNoiseTurn')
                omega = omega + randn() * 0.01;
            end
            cos_w = cos(omega * model.T);
            sin_w = sin(omega * model.T);
            prev_state = sensorTrajectories{s}(:, t-1);
            prev_pos = prev_state(1:2);
            prev_vel = prev_state(3:4);
            new_vel = [cos_w * prev_vel(1) - sin_w * prev_vel(2);
                       sin_w * prev_vel(1) + cos_w * prev_vel(2)];
            new_pos = prev_pos + new_vel * model.T;
            sensorTrajectories{s}(:, t) = [new_pos; new_vel] + sampleSensorProcessNoise(model, s);
        end
    end
end

function sensorTrajectories = generateFormationSensorTrajectories(model, simulationLength)
% GENERATEFORMATIONSENSORTRAJECTORIES - Rigid formation sensor motion
    sensorTrajectories = cell(1, model.numberOfSensors);
    if ~isfield(model, 'sensorFormationType')
        model.sensorFormationType = 'Triangle';
    end
    if ~isfield(model, 'sensorFormationSpacing')
        model.sensorFormationSpacing = 20;
    end
    if ~isfield(model, 'sensorFormationCenterStart')
        model.sensorFormationCenterStart = [-80; 0];
    end
    if ~isfield(model, 'sensorFormationVelocity')
        model.sensorFormationVelocity = [0.8; 0];
    end
    offsets = computeFormationOffsets(model.sensorFormationType, model.sensorFormationSpacing, model.numberOfSensors);
    center_state = [model.sensorFormationCenterStart; model.sensorFormationVelocity];
    for t = 1:simulationLength
        if t > 1
            center_state = model.A * center_state;
        end
        for s = 1:model.numberOfSensors
            offset = offsets(:, s);
            sensorTrajectories{s}(:, t) = [center_state(1:2) + offset; center_state(3:4)];
        end
    end
end

function offsets = computeFormationOffsets(formationType, spacing, count)
% COMPUTEFORMATIONOFFSETS - Relative offsets for formation geometry
    if nargin < 3
        count = 3;
    end
    if nargin < 2
        spacing = 15;
    end
    if nargin < 1 || isempty(formationType)
        formationType = 'Triangle';
    end
    formationType = upper(formationType);
    switch formationType
        case 'TRIANGLE'
            radius = spacing / sqrt(3);
            angles = (0:2) * 2 * pi / 3;
            baseOffsets = radius * [cos(angles); sin(angles)];
        case 'LINE'
            idx = (0:count-1) - (count-1)/2;
            baseOffsets = [zeros(1, count); idx * spacing];
        case {'LEADER4','ONEPLUSFOUR'}
            % V-shape (chevron): leader at the tip, two followers per row
            y1 = 0.7 * spacing;
            y2 = 1.4 * spacing;
            baseOffsets = [0, -spacing, -spacing, -2*spacing, -2*spacing;
                           0,  y1,       -y1,       y2,        -y2];
        case {'LEADER3','ONEPLUSTHREE'}
            % V-shape: two on first row, one centered on second row
            y1 = 0.7 * spacing;
            baseOffsets = [0, -spacing, -spacing, -2*spacing;
                           0,  y1,       -y1,       0];
        case 'CIRCLE'
            angles = 2 * pi * (0:count-1) / max(count, 1);
            baseOffsets = spacing * [cos(angles); sin(angles)];
        otherwise
            baseOffsets = zeros(2, max(count, 1));
    end
    if size(baseOffsets, 2) >= count
        offsets = baseOffsets(:, 1:count);
    else
        reps = ceil(count / size(baseOffsets, 2));
        offsets = repmat(baseOffsets, 1, reps);
        offsets = offsets(:, 1:count);
    end
end

function enabled = isFovEnabled(model)
    enabled = isfield(model, 'sensorFovEnabled') && model.sensorFovEnabled && ...
        isfield(model, 'sensorFovHalfAngleDeg') && isfield(model, 'sensorFovRange');
end

function value = getFovValue(model, fieldName, sensorIdx, defaultValue)
    if isfield(model, fieldName)
        v = model.(fieldName);
        if isempty(v)
            value = defaultValue;
        elseif isscalar(v)
            value = v;
        elseif numel(v) >= sensorIdx
            value = v(sensorIdx);
        else
            value = v(1);
        end
    else
        value = defaultValue;
    end
end

function inFov = isTargetInFov(sensorPos, sensorVel, targetPos, halfAngleDeg, maxRange)
    rel = targetPos - sensorPos;
    dist = norm(rel);
    if ~(isfinite(maxRange))
        maxRange = inf;
    end
    if dist > maxRange
        inFov = false;
        return;
    end
    speed = norm(sensorVel);
    if speed < 1e-6 || dist < 1e-9
        inFov = true;
        return;
    end
    cosTheta = (rel' * sensorVel) / (dist * speed);
    cosLimit = cosd(halfAngleDeg);
    inFov = cosTheta >= cosLimit;
end

function noise = sampleSensorProcessNoise(model, s)
% SAMPLE SENSOR PROCESS NOISE - Robustly sample even if covariance is PSD/zero
    noise = zeros(4, 1);
    if ~isfield(model, 'sensorProcessNoise') || numel(model.sensorProcessNoise) < s
        return;
    end
    Q = model.sensorProcessNoise{s};
    if isempty(Q) || ~any(Q(:))
        return;
    end
    Q = (Q + Q') * 0.5;
    [L, p] = chol(Q, 'lower');
    if p > 0
        [L, p] = chol(Q + 1e-9 * eye(size(Q)), 'lower');
        if p > 0
            return;
        end
    end
    noise = L * randn(4, 1);
end
