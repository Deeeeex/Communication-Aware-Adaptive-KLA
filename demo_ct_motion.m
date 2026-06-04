% DEMO_CT_MOTION - Demonstrate Coordinated Turn (CT) sensor motion
%   Shows Phase 2 enhanced sensor motion capabilities

clear; clc;
setPath;

%% Configuration
numberOfSensors = 3;
clutterRates = [5, 5, 5];
detectionProbabilities = [0.9, 0.9, 0.9];
q = [3, 3, 3];

%% CT Motion Configuration
sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CT'; % Coordinated Turn model
sensorMotionConfig.processNoiseStd = 0.1;
sensorMotionConfig.turnRate = 0.05; % Turn rate (rad/step)
sensorMotionConfig.turnModel = 'WhiteNoiseTurn';

%% Sensor initial states
sensorInitialStates = cell(1, numberOfSensors);
sensorInitialPositions = [-50, 0; 50, 0; 0, 70];
sensorInitialVelocities = [0.3, 0; -0.3, 0; 0, -0.2];

for s = 1:numberOfSensors
    sensorInitialStates{s} = [sensorInitialPositions(s, :)'; sensorInitialVelocities(s, :)'];
end
sensorMotionConfig.initialStates = sensorInitialStates;

%% Generate model with CT motion
model = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'GA', 'LBP', 'Fixed', sensorMotionConfig);

%% Configure enhanced model for CT motion
model.sensorMotionType = 'CT';
model.sensorTurnRate = sensorMotionConfig.turnRate;
model.sensorTurnModel = sensorMotionConfig.turnModel;

%% Generate sensor trajectories and measurements
fprintf('=====================================\n');
fprintf('CT Motion Model Demonstration\n');
fprintf('=====================================\n');
fprintf('Sensors: %d\n', numberOfSensors);
fprintf('Turn rate: %.3f rad/step\n', sensorMotionConfig.turnRate);
fprintf('Process noise std: %.3f\n', sensorMotionConfig.processNoiseStd);
fprintf('\n');

[groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);

%% Run filter
fprintf('Running LMB filter with CT motion sensors...\n');
stateEstimates = runParallelUpdateLmbFilter(model, measurements, [], sensorTrajectories);

%% Compute OSPA
[eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

fprintf('Average E-OSPA: %.3f\n', mean(eOspa));
fprintf('Average H-OSPA: %.3f\n', mean(hOspa));

%% Visualization
fprintf('Generating visualization...\n');

figure('Position', [100, 100, 1400, 600]);

%% Subplot 1: Sensor trajectories (CT motion)
subplot(2, 3, 1);
hold on;
colors = {'b', 'r', 'g', 'm'};
markers = {'o', 's', '^', 'd'};

for s = 1:numberOfSensors
    sensorTraj = sensorTrajectories{s};
    plot(sensorTraj(1, :), sensorTraj(2, :), ...
        '--', 'Color', colors{s}, 'LineWidth', 1.5);
    plot(sensorTraj(1, 1), sensorTraj(2, 1), ...
        markers{s}, 'Color', colors{s}, 'MarkerSize', 10, ...
        'MarkerFaceColor', colors{s});
    plot(sensorTraj(1, end), sensorTraj(2, end), ...
        markers{s}, 'Color', colors{s}, 'MarkerSize', 10, ...
        'MarkerFaceColor', colors{s});
end

xlabel('X Position', 'FontSize', 10);
ylabel('Y Position', 'FontSize', 10);
title(sprintf('CT Sensor Motion (Turn Rate=%.3f)', sensorMotionConfig.turnRate), 'FontSize', 12);
grid on;
axis equal;
legend('Sensor Trajectories', 'Location', 'best');

%% Subplot 2: Ground truth
subplot(2, 3, 2);
hold on;

for i = 1:length(groundTruth)
    traj = groundTruth{i};
    colorIdx = mod(i-1, length(colors)) + 1;
    xTraj = traj(2, :);
    yTraj = traj(3, :);
    plot(xTraj, yTraj, '-', 'Color', colors{colorIdx}, 'LineWidth', 1.5);
end

xlabel('X Position', 'FontSize', 10);
ylabel('Y Position', 'FontSize', 10);
title('Ground Truth Trajectories', 'FontSize', 12);
grid on;
axis equal;

%% Subplot 3: Estimated trajectories
subplot(2, 3, 3);
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

xlabel('X Position', 'FontSize', 10);
ylabel('Y Position', 'FontSize', 10);
title(sprintf('Estimated Trajectories (E-OSPA=%.3f)', mean(eOspa)), 'FontSize', 12);
grid on;
axis equal;

%% Subplot 4: OSPA over time
subplot(2, 3, 4);
hold on;

timeSteps = 1:length(eOspa);
plot(timeSteps, eOspa, 'b-', 'LineWidth', 2);
plot(timeSteps, hOspa, 'r--', 'LineWidth', 2);

xlabel('Time Step', 'FontSize', 10);
ylabel('OSPA Distance', 'FontSize', 10);
title('OSPA Over Time', 'FontSize', 12);
legend('E-OSPA', 'H-OSPA', 'Location', 'best');
grid on;

%% Subplot 5: Sensor velocity analysis
subplot(2, 3, 5);
hold on;

for s = 1:numberOfSensors
    sensorTraj = sensorTrajectories{s};
    velX = sensorTraj(3, :);
    velY = sensorTraj(4, :);

    % Calculate speed over time
    speed = sqrt(velX.^2 + velY.^2);
    plot(1:length(speed), speed, ['-' colors{s}], 'LineWidth', 1.5);
end

xlabel('Time Step', 'FontSize', 10);
ylabel('Sensor Speed', 'FontSize', 10);
title('Sensor Speed Over Time', 'FontSize', 12);
grid on;
legend({'Sensor 1', 'Sensor 2', 'Sensor 3'}, 'Location', 'best');

%% Subplot 6: Cardinality error
subplot(2, 3, 6);
hold on;

estimatedCardinality = cellfun(@(x) size(x, 2), stateEstimates.labels);
trueCardinality = cellfun(@(x) length(x), groundTruthRfs.x);
cardError = abs(estimatedCardinality - trueCardinality);

plot(1:length(cardError), cardError, 'b-', 'LineWidth', 2);
xlabel('Time Step', 'FontSize', 10);
ylabel('Cardinality Error', 'FontSize', 10);
title('Cardinality Error Over Time', 'FontSize', 12);
grid on;

fprintf('\nDone!\n');
fprintf('Visualization shows:\n');
fprintf('  - Top left: CT sensor trajectories\n');
fprintf('  - Top center: Ground truth\n');
fprintf('  - Top right: Estimated trajectories\n');
fprintf('  - Bottom left: OSPA over time\n');
fprintf('  - Bottom center: Sensor speeds\n');
fprintf('  - Bottom right: Cardinality error\n');
