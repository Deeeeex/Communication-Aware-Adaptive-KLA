% RUNMULTISENSORFILTERS_MOBILE - Run multi-sensor LMB filters with mobile sensors
%   演示移动传感器支持的最小多传感器流程。
%
% 文件导读：
%   移动传感器小型示例。它保留多传感器滤波主线，同时把重点放在
%   sensor trajectory 和相对测量几何上。阅读时先确认传感器运动配置，
%   再看观测生成和滤波调用；大规模 formation 脚本是在此基础上扩展。

%% Admin
close all; clc;
setPath;

%% 1. 传感器基础配置
numberOfSensors = 3;
clutterRates = [5 5 5];
detectionProbabilities = [0.67 0.70 0.73];
q = [4 3 2];

%% 2. 传感器运动配置：这里使用常速度模型
sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV'; % Constant Velocity model
sensorMotionConfig.processNoiseStd = 0.1; % Sensor motion process noise

%% 3. 初始位置和速度：每行对应一个传感器
sensorInitialStates = cell(1, numberOfSensors);
sensorInitialPositions = [-50, 0, 0.5, 0;
                         50, 0, -0.5, 0;
                          0, 70, 0, -0.2];

for s = 1:numberOfSensors
    sensorInitialStates{s} = sensorInitialPositions(s, :)';
end

sensorMotionConfig.initialStates = sensorInitialStates;

%% Generate model with mobile sensors
model = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'GA', 'LBP', 'Fixed', sensorMotionConfig);

%% Display configuration
fprintf('=====================================\n');
fprintf('Mobile Sensor Configuration\n');
fprintf('=====================================\n');
fprintf('Number of sensors: %d\n', numberOfSensors);
fprintf('Sensor motion enabled: %s\n', mat2str(model.sensorMotionEnabled));
fprintf('Motion model: %s\n', sensorMotionConfig.motionType);
fprintf('Process noise std: %.3f\n', sensorMotionConfig.processNoiseStd);
fprintf('\nSensor Initial States:\n');
for s = 1:numberOfSensors
    fprintf('  Sensor %d: Pos=[%.1f, %.1f], Vel=[%.2f, %.2f]\n', ...
        s, sensorInitialPositions(s, 1), sensorInitialPositions(s, 2), ...
        sensorInitialPositions(s, 3), sensorInitialPositions(s, 4));
end
fprintf('\n');

%% Adaptive fusion weights (GA/AA)
model.adaptiveFusion = struct();
model.adaptiveFusion.enabled = false;
model.adaptiveFusion.emaAlpha = 0.7;
model.adaptiveFusion.minWeight = 0.05;

%% Generate observations
[groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);

%% Display sensor trajectory info
fprintf('Sensor Trajectory Summary:\n');
for s = 1:numberOfSensors
    startPos = sensorTrajectories{s}(:, 1)';
    endPos = sensorTrajectories{s}(:, end)';
    distTravelled = norm(endPos(1:2) - startPos(1:2));
    fprintf('  Sensor %d: Start=[%.1f, %.1f], End=[%.1f, %.1f], Distance=%.2f\n', ...
        s, startPos(1), startPos(2), endPos(1), endPos(2), distTravelled);
end
fprintf('\n');

%% Run filters
filterType = 'PU'; % 'IC', 'PU', 'LMBM'

if(strcmp(filterType, 'IC'))
    stateEstimates = runIcLmbFilter(model, measurements, sensorTrajectories);
elseif(strcmp(filterType, 'PU'))
    stateEstimates = runParallelUpdateLmbFilter(model, measurements, [], sensorTrajectories);
else
    stateEstimates = runMultisensorLmbmFilter(model, measurements);
end

%% Compute OSPA scores
[eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

fprintf('=====================================\n');
fprintf('Performance Metrics\n');
fprintf('=====================================\n');
fprintf('Average E-OSPA: %.3f\n', mean(eOspa));
fprintf('Average H-OSPA: %.3f\n', mean(hOspa));
fprintf('\n');

%% Plotting
fprintf('Generating visualization...\n');

% Create figure for sensor trajectories and tracking results
figure('Position', [100, 100, 1200, 500]);

% Plot ground truth trajectories
subplot(1, 2, 1);
hold on;
colors = {'b', 'r', 'g', 'm', 'c'};
markers = {'o', 's', '^', 'd', 'v'};

for i = 1:length(groundTruth)
    colorIdx = mod(i-1, length(colors)) + 1;
    traj = groundTruth{i};
    xTraj = traj(2, :);
    yTraj = traj(3, :);
    plot(xTraj, yTraj, '-', 'Color', colors{colorIdx}, 'LineWidth', 1.5);
    plot(xTraj(1), yTraj(1), ...
        'Marker', markers{colorIdx}, 'MarkerSize', 8, ...
        'MarkerFaceColor', colors{colorIdx}, 'Color', colors{colorIdx});
end

% Plot sensor trajectories
if model.sensorMotionEnabled
    for s = 1:numberOfSensors
        sensorTraj = sensorTrajectories{s};
        plot(sensorTraj(1, :), sensorTraj(2, :), ...
            'k--', 'LineWidth', 2, 'Color', 'k');
        plot(sensorTraj(1, 1), sensorTraj(2, 1), ...
            'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'Color', 'k');
        plot(sensorTraj(1, end), sensorTraj(2, end), ...
            'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'Color', 'k');
    end
end

xlabel('X Position', 'FontSize', 10);
ylabel('Y Position', 'FontSize', 10);
title('Ground Truth Trajectories and Sensor Paths', 'FontSize', 12);
legend('Ground Truth', 'Location', 'best');
grid on;
axis equal;

% Plot estimated trajectories
subplot(1, 2, 2);
hold on;

for i = 1:length(stateEstimates.objects)
    obj = stateEstimates.objects(i);
    if obj.trajectoryLength > 0
        validIdx = 1:obj.trajectoryLength;
        traj = [obj.timestamps(validIdx); obj.trajectory(1:2, validIdx)];
        colorIdx = mod(i-1, length(colors)) + 1;
        xTraj = traj(2, :);
        yTraj = traj(3, :);
        plot(xTraj, yTraj, '-', 'Color', colors{colorIdx}, 'LineWidth', 1.5);
    end
end

% Plot sensor trajectories
if model.sensorMotionEnabled
    for s = 1:numberOfSensors
        sensorTraj = sensorTrajectories{s};
        plot(sensorTraj(1, :), sensorTraj(2, :), ...
            'k--', 'LineWidth', 2);
        plot(sensorTraj(1, 1), sensorTraj(2, 1), ...
            'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    end
end

xlabel('X Position', 'FontSize', 10);
ylabel('Y Position', 'FontSize', 10);
title(sprintf('Estimated Trajectories (GA Fusion, E-OSPA=%.3f)', mean(eOspa)), 'FontSize', 12);
grid on;
axis equal;

fprintf('Done!\n');
