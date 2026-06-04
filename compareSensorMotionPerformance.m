function compareSensorMotionPerformance(varargin)
% COMPARESENSORMOTIONPERFORMANCE - Compare static vs mobile sensor performance
%   compareSensorMotionPerformance()
%
%   Compares tracking performance between static and mobile sensors across different
%   motion configurations and noise levels.
%
%   Name-Value Parameters:
%       'MotionTypes' - Cell array of motion types to test ['CV', 'CT', 'Formation']
%       'NoiseLevels' - Array of process noise standard deviations
%       'NumberOfTrials' - Number of Monte Carlo trials [default: 10]
%       'NumberOfSensors' - Number of sensors [default: 3]
%       'GeneratePlots' - Generate comparison plots [default: true]
%
%   Outputs:
%       results - Struct containing performance comparison data

%% Input parsing
defaultMotionTypes = {'Static', 'CV', 'CT'};
defaultNoiseLevels = [0.05, 0.1, 0.2, 0.5];
defaultNumberOfTrials = 5;
defaultNumberOfSensors = 3;
defaultGeneratePlots = true;

p = inputParser;
addParameter(p, 'MotionTypes', defaultMotionTypes, @iscell);
addParameter(p, 'NoiseLevels', defaultNoiseLevels, @isnumeric);
addParameter(p, 'NumberOfTrials', defaultNumberOfTrials, @isnumeric);
addParameter(p, 'NumberOfSensors', defaultNumberOfSensors, @isnumeric);
addParameter(p, 'GeneratePlots', defaultGeneratePlots, @islogical);

parse(p, varargin{:});
params = p.Results;

%% Initialize results
nMotionTypes = length(params.MotionTypes);
nNoiseLevels = length(params.NoiseLevels);

results = struct();
results.motionTypes = params.MotionTypes;
results.noiseLevels = params.NoiseLevels;
results.eOspa = zeros(nMotionTypes, nNoiseLevels, params.NumberOfTrials);
results.hOspa = zeros(nMotionTypes, nNoiseLevels, params.NumberOfTrials);
results.runtime = zeros(nMotionTypes, nNoiseLevels, params.NumberOfTrials);

%% Base configuration
numberOfSensors = params.NumberOfSensors;
clutterRates = ones(1, numberOfSensors) * 5;
detectionProbabilities = ones(1, numberOfSensors) * 0.9;
q = ones(1, numberOfSensors) * 3;

%% Sensor initial positions
sensorInitialPositions = [-50, 0; 50, 0; 0, 70];
sensorInitialVelocities = [0.5, 0; -0.5, 0; 0, -0.2];
sensorInitialStates = cell(1, numberOfSensors);
for s = 1:numberOfSensors
    sensorInitialStates{s} = [sensorInitialPositions(s, :); sensorInitialVelocities(s, :)]';
end

%% Main evaluation loop
fprintf('=====================================\n');
fprintf('Sensor Motion Performance Comparison\n');
fprintf('=====================================\n');
fprintf('Motion types: %s\n', strjoin(params.MotionTypes, ', '));
fprintf('Noise levels: [%s]\n', num2str(params.NoiseLevels));
fprintf('Trials per configuration: %d\n', params.NumberOfTrials);
fprintf('\n');

for m = 1:nMotionTypes
    motionType = params.MotionTypes{m};
    fprintf('Testing motion type: %s (%d/%d)\n', motionType, m, nMotionTypes);

    for n = 1:nNoiseLevels
        noiseStd = params.NoiseLevels(n);
        fprintf('  Noise level %.3f (%d/%d)...', noiseStd, n, nNoiseLevels);

        for trial = 1:params.NumberOfTrials
            if mod(trial, max(1, floor(params.NumberOfTrials / 3))) == 0
                fprintf('.');
            end

            %% Configure sensor motion
            sensorMotionConfig = struct();

            if strcmp(motionType, 'Static')
                sensorMotionConfig.enabled = false;
            else
                sensorMotionConfig.enabled = true;
                sensorMotionConfig.motionType = motionType;
                sensorMotionConfig.processNoiseStd = noiseStd;
                sensorMotionConfig.initialStates = sensorInitialStates;

                % Additional parameters for CT model
                if strcmp(motionType, 'CT')
                    sensorMotionConfig.turnRate = 0.05;
                    sensorMotionConfig.turnModel = 'ConstantTurn';
                end
            end

            %% Generate model and data
            model = generateMultisensorModel(numberOfSensors, clutterRates, ...
                detectionProbabilities, q, 'GA', 'LBP', 'Fixed', sensorMotionConfig);

            [groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);

            %% Run filter
            tic;
            stateEstimates = runParallelUpdateLmbFilter(model, measurements);
            runtime = toc;

            %% Compute OSPA
            [eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);

            %% Store results
            results.eOspa(m, n, trial) = mean(eOspa);
            results.hOspa(m, n, trial) = mean(hOspa);
            results.runtime(m, n, trial) = runtime;
        end
        fprintf(' done\n');
    end
    fprintf('\n');
end

%% Compute statistics
results.meanEospa = squeeze(mean(results.eOspa, 3));
results.stdEospa = squeeze(std(results.eOspa, 0, 3));
results.meanHospa = squeeze(mean(results.hOspa, 3));
results.meanRuntime = squeeze(mean(results.runtime, 3));

%% Display summary
fprintf('=====================================\n');
fprintf('Performance Summary\n');
fprintf('=====================================\n');

for m = 1:nMotionTypes
    fprintf('\n--- %s Sensors ---\n', params.MotionTypes{m});
    fprintf('Noise | E-OSPA | H-OSPA | Runtime\n');
    fprintf('-------+---------+---------+---------\n');

    for n = 1:nNoiseLevels
        fprintf(' %.3f | %7.3f | %7.3f | %8.3f\n', ...
            params.NoiseLevels(n), ...
            results.meanEospa(m, n), ...
            results.meanHospa(m, n), ...
            results.meanRuntime(m, n));
    end
end

fprintf('\n');

%% Generate plots
if params.GeneratePlots
    generateComparisonPlots(results, params);
end

end

%% Helper function to generate comparison plots
function generateComparisonPlots(results, params)
    nMotionTypes = length(params.MotionTypes);
    nNoiseLevels = length(params.NoiseLevels);

    %% Plot 1: E-OSPA comparison
    figure('Position', [100, 100, 900, 600]);
    hold on;
    colors = {'b', 'r', 'g', 'm'};
    markers = {'o', 's', '^', 'd'};

    for m = 1:nMotionTypes
        colorIdx = mod(m-1, length(colors)) + 1;
        markerIdx = mod(m-1, length(markers)) + 1;
        errorbar(params.NoiseLevels, results.meanEospa(m, :), ...
            results.stdEospa(m, :), ...
            ['-' markers{markerIdx}], 'Color', colors{colorIdx}, ...
            'LineWidth', 2, 'MarkerSize', 8, ...
            'MarkerFaceColor', colors{colorIdx});
    end

    xlabel('Sensor Motion Noise (std)', 'FontSize', 12);
    ylabel('E-OSPA Distance', 'FontSize', 12);
    title('E-OSPA Performance vs Sensor Motion Noise', 'FontSize', 14);
    grid on;
    legend(params.MotionTypes, 'Location', 'best');

    %% Plot 2: Runtime comparison
    figure('Position', [200, 200, 900, 600]);
    hold on;

    for m = 1:nMotionTypes
        colorIdx = mod(m-1, length(colors)) + 1;
        markerIdx = mod(m-1, length(markers)) + 1;
        plot(params.NoiseLevels, results.meanRuntime(m, :), ...
            ['-' markers{markerIdx}], 'Color', colors{colorIdx}, ...
            'LineWidth', 2, 'MarkerSize', 8, ...
            'MarkerFaceColor', colors{colorIdx});
    end

    xlabel('Sensor Motion Noise (std)', 'FontSize', 12);
    ylabel('Runtime (seconds)', 'FontSize', 12);
    title('Runtime vs Sensor Motion Noise', 'FontSize', 14);
    grid on;
    legend(params.MotionTypes, 'Location', 'best');

    %% Plot 3: Heatmap
    figure('Position', [300, 300, 1000, 600]);

    for m = 1:nMotionTypes
        subplot(1, nMotionTypes, m);
        imagesc(results.meanEospa(m, :));
        colorbar;
        xlabel('Noise Level Index', 'FontSize', 10);
        ylabel('Noise Level', 'FontSize', 10);
        title(sprintf('%s Sensors E-OSPA', params.MotionTypes{m}), 'FontSize', 12);

        set(gca, 'XTick', 1:length(params.NoiseLevels));
        set(gca, 'YTick', 1:length(params.NoiseLevels));
        set(gca, 'XTickLabel', arrayfun(@num2str, params.NoiseLevels, 'UniformOutput', false));
    end

end
