% TEST_MOBILE_SENSORS - Simple test script for mobile sensor functionality
%   Tests Phase 1 basic framework implementation

clear; clc;
setPath;

%% Test 1: Static sensors (baseline)
fprintf('Test 1: Static Sensors (Baseline)\n');
fprintf('=====================================\n');

numberOfSensors = 2;
clutterRates = [5, 5];
detectionProbabilities = [0.9, 0.9];
q = [3, 3];

% Disable sensor motion
sensorMotionConfig = struct();
sensorMotionConfig.enabled = false;

model_static = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'GA', 'LBP', 'Fixed', sensorMotionConfig);

fprintf('Sensor motion enabled: %s\n', mat2str(model_static.sensorMotionEnabled));

[groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model_static);
stateEstimates = runParallelUpdateLmbFilter(model_static, measurements);
[eOspa, hOspa] = computeSimulationOspa(model_static, groundTruthRfs, stateEstimates);

fprintf('E-OSPA: %.3f\n', mean(eOspa));
fprintf('H-OSPA: %.3f\n\n', mean(hOspa));

%% Test 2: Mobile sensors with CV motion
fprintf('Test 2: Mobile Sensors with CV Motion\n');
fprintf('=====================================\n');

% Enable sensor motion
sensorMotionConfig = struct();
sensorMotionConfig.enabled = true;
sensorMotionConfig.motionType = 'CV';
sensorMotionConfig.processNoiseStd = 0.1;

% Configure sensor initial states
sensorInitialStates = cell(1, numberOfSensors);
sensorInitialStates{1} = [-50; 0; 0.5; 0];
sensorInitialStates{2} = [50; 0; -0.5; 0];
sensorMotionConfig.initialStates = sensorInitialStates;

model_mobile = generateMultisensorModel(numberOfSensors, clutterRates, ...
    detectionProbabilities, q, 'GA', 'LBP', 'Fixed', sensorMotionConfig);

fprintf('Sensor motion enabled: %s\n', mat2str(model_mobile.sensorMotionEnabled));
fprintf('Motion type: %s\n', model_mobile.sensorMotionType);
fprintf('Process noise std: %.3f\n', model_mobile.sensorProcessNoiseStd);

[groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model_mobile);

if ~isempty(sensorTrajectories)
    fprintf('\nSensor Trajectories:\n');
    for s = 1:numberOfSensors
        start = sensorTrajectories{s}(:, 1)';
        final = sensorTrajectories{s}(:, end)';
        distance = norm(final(1:2) - start(1:2));
        fprintf('  Sensor %d: (%.1f, %.1f) -> (%.1f, %.1f), distance=%.2f\n', ...
            s, start(1), start(2), final(1), final(2), distance);
    end
end

stateEstimates = runParallelUpdateLmbFilter(model_mobile, measurements);
[eOspa, hOspa] = computeSimulationOspa(model_mobile, groundTruthRfs, stateEstimates);

fprintf('E-OSPA: %.3f\n', mean(eOspa));
fprintf('H-OSPA: %.3f\n\n', mean(hOspa));

%% Test 3: Different sensor motion parameters
fprintf('Test 3: Varying Sensor Motion Noise\n');
fprintf('=====================================\n');

noiseLevels = [0.05, 0.2, 0.5];
ospa_results = zeros(length(noiseLevels), 1);

for i = 1:length(noiseLevels)
    sensorMotionConfig.processNoiseStd = noiseLevels(i);
    model = generateMultisensorModel(numberOfSensors, clutterRates, ...
        detectionProbabilities, q, 'GA', 'LBP', 'Fixed', sensorMotionConfig);
    
    [groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
    stateEstimates = runParallelUpdateLmbFilter(model, measurements);
    [eOspa, ~] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
    
    ospa_results(i) = mean(eOspa);
    fprintf('Noise std=%.3f: E-OSPA=%.3f\n', noiseLevels(i), ospa_results(i));
end

fprintf('\n=====================================\n');
fprintf('All tests completed successfully!\n');
