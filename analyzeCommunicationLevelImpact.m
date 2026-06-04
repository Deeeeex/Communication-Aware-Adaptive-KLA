function commImpactResults = analyzeCommunicationLevelImpact(varargin)
% ANALYZECOMMUNICATIONLEVELIMPACT - Analyze impact of commConfig.level on GA/AA fusion performance
%   commImpactResults = analyzeCommunicationLevelImpact() runs analysis with default parameters
%
%   This script analyzes how different communication constraint levels (0, 1, 2, 3)
%   affect the performance of GA and AA multi-sensor fusion algorithms.
%
%   Communication Levels:
%       0 - Ideal communication (no constraints)
%       1 - Global bandwidth limit
%       2 - Bandwidth + link loss
%       3 - Bandwidth + link loss + node outage
%
%   Name-Value Parameters:
%       'CommLevels'           - Array of levels to test [default: [0, 1, 2, 3]]
%       'FusionModes'          - Cell array of modes to test [default: {'GA', 'AA'}]
%       'NumberOfTrials'       - Number of Monte Carlo trials [default: 10]
%       'NumberOfSensors'      - Number of sensors [default: 3]
%       'ClutterRates'         - Per-sensor clutter rates [default: [5 5 5]]
%       'DetectionProbabilities' - Per-sensor detection probs [default: [0.67 0.70 0.73]]
%       'SensorPrecision'      - Measurement precision q [default: [4 3 2]]
%       'GlobalMaxMeasurements' - Max measurements per step [default: 25]
%       'Verbose'              - Display progress [default: true]
%       'GeneratePlots'         - Generate plots [default: true]
%
%   Output:
%       commImpactResults - Struct containing comprehensive analysis results
%
%   Examples:
%       results = analyzeCommunicationLevelImpact('NumberOfTrials', 20);
%
%       results = analyzeCommunicationLevelImpact('CommLevels', [0, 2], ...
%                                                 'FusionModes', {'GA'});

%% Input parsing and validation
defaultCommLevels = [0, 1, 2, 3];
defaultFusionModes = {'GA', 'AA'};
defaultEnableAdaptiveFusion = true;
defaultNumberOfTrials = 10;
defaultNumberOfSensors = 3;
defaultClutterRates = [5, 5, 5];
defaultDetectionProbs = [0.67, 0.70, 0.73];
defaultSensorPrecision = [4, 3, 2];
defaultGlobalMaxMeasurements = 25;
defaultVerbose = true;
defaultGeneratePlots = true;

p = inputParser;
addParameter(p, 'CommLevels', defaultCommLevels, @(x) isnumeric(x) && all(ismember(x, [0:3])));
addParameter(p, 'FusionModes', defaultFusionModes, @iscell);
addParameter(p, 'EnableAdaptiveFusion', defaultEnableAdaptiveFusion, @islogical);
addParameter(p, 'NumberOfTrials', defaultNumberOfTrials, @(x) isnumeric(x) && x > 0);
addParameter(p, 'NumberOfSensors', defaultNumberOfSensors, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ClutterRates', defaultClutterRates, @isnumeric);
addParameter(p, 'DetectionProbabilities', defaultDetectionProbs, @isnumeric);
addParameter(p, 'SensorPrecision', defaultSensorPrecision, @isnumeric);
addParameter(p, 'GlobalMaxMeasurements', defaultGlobalMaxMeasurements, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Verbose', defaultVerbose, @islogical);
addParameter(p, 'GeneratePlots', defaultGeneratePlots, @islogical);

parse(p, varargin{:});
params = p.Results;

%% Initialize results structure
commImpactResults = struct();
commImpactResults.metadata = struct();
commImpactResults.metadata.timestamp = datetime('now');
commImpactResults.metadata.parameters = params;
commImpactResults.metadata.matlabVersion = version;

%% Expand fusion modes if adaptive fusion is enabled
if params.EnableAdaptiveFusion
    fusionModesList = {};
    for i = 1:length(params.FusionModes)
        fusionModesList{end+1} = params.FusionModes{i};
        fusionModesList{end+1} = [params.FusionModes{i} '-Adaptive'];
    end
    actualFusionModes = fusionModesList;
else
    actualFusionModes = params.FusionModes;
end

nCommLevels = length(params.CommLevels);
nFusionModes = length(actualFusionModes);
nTrials = params.NumberOfTrials;

%% Initialize storage arrays
ospaScores = zeros(nFusionModes, nCommLevels, nTrials);
cardinalityError = zeros(nFusionModes, nCommLevels, nTrials);
runtimes = zeros(nFusionModes, nCommLevels, nTrials);

%% Communication statistics storage
totalDrops = zeros(nFusionModes, nCommLevels, nTrials);
avgDeliveryRatio = zeros(nFusionModes, nCommLevels, nTrials);
outageCount = zeros(nFusionModes, nCommLevels, nTrials);

%% Main analysis loop
if params.Verbose
    fprintf('=====================================\n');
    fprintf('Communication Level Impact Analysis\n');
    fprintf('=====================================\n');
    fprintf('Parameters:\n');
    fprintf('  Communication levels: [%s]\n', num2str(params.CommLevels));
    fprintf('  Fusion modes: %s\n', strjoin(params.FusionModes, ', '));
    fprintf('  Number of trials: %d\n', nTrials);
    fprintf('  Sensors: %d\n', params.NumberOfSensors);
    fprintf('\n');
end

for f = 1:nFusionModes
    fusionModeName = actualFusionModes{f};
    
    %% Parse fusion mode and adaptive flag
    if endsWith(fusionModeName, '-Adaptive')
        fusionMode = fusionModeName(1:end-9); % Remove '-Adaptive' suffix
        useAdaptiveFusion = true;
    else
        fusionMode = fusionModeName;
        useAdaptiveFusion = false;
    end
    
    if params.Verbose
        fprintf('=== Testing fusion mode: %s ===\n', fusionModeName);
    end
    
    for l = 1:nCommLevels
        commLevel = params.CommLevels(l);
        
        if params.Verbose
            fprintf('  Communication level %d: ', commLevel);
        end
        
        for trial = 1:nTrials
            if params.Verbose && mod(trial, 5) == 0
                fprintf('.');
            end
            
            %% Generate model
            model = generateMultisensorModel(params.NumberOfSensors, ...
                params.ClutterRates, ...
                params.DetectionProbabilities, ...
                params.SensorPrecision, ...
                fusionMode, 'LBP', 'Fixed');
            
            %% Configure adaptive fusion
            model.adaptiveFusion = struct();
            model.adaptiveFusion.enabled = useAdaptiveFusion;
            model.adaptiveFusion.emaAlpha = 0.7;
            model.adaptiveFusion.minWeight = 0.05;
            
            %% Generate observations
            [groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
            
            %% Apply communication model
            commConfig = struct();
            commConfig.level = commLevel;
            commConfig.globalMaxMeasurementsPerStep = params.GlobalMaxMeasurements;
            commConfig.sensorWeights = ones(1, params.NumberOfSensors) / params.NumberOfSensors;
            commConfig.priorityPolicy = 'weightedPriority';
            commConfig.measurementSelectionPolicy = 'random';
            commConfig.linkModel = 'fixed';
            commConfig.pDrop = 0.2;
            commConfig.maxOutageNodes = 1;
            
            [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
            
            %% Run filter
            tic;
            stateEstimates = runParallelUpdateLmbFilter(model, measurementsDelivered, commStats);
            runtime = toc;
            
            %% Compute performance metrics
            [eOspa, hOspa] = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
            ospa = eOspa;
            
            %% Cardinality error
            estimatedCardinality = cellfun(@(x) size(x, 2), stateEstimates.labels);
            trueCardinality = cellfun(@(x) size(x, 2), groundTruthRfs.x);
            cardError = mean(abs(estimatedCardinality(:) - trueCardinality(:)));
            
            %% Store results
            ospaScores(f, l, trial) = mean(ospa);
            cardinalityError(f, l, trial) = cardError;
            runtimes(f, l, trial) = runtime;
            
            %% Store communication statistics
            if isfield(commStats, 'totalDropped')
                totalDrops(f, l, trial) = commStats.totalDropped;
            end
            if isfield(commStats, 'avgDeliveryRatio')
                avgDeliveryRatio(f, l, trial) = commStats.avgDeliveryRatio;
            end
            if isfield(commStats, 'outageEvents')
                outageCount(f, l, trial) = length(commStats.outageEvents);
            end
        end
        
        if params.Verbose
            fprintf(' done\n');
        end
    end
end

%% Organize results
commImpactResults.ospa = squeeze(mean(ospaScores, 3));
commImpactResults.ospaStd = squeeze(std(ospaScores, 0, 3));
commImpactResults.cardinalityError = squeeze(mean(cardinalityError, 3));
commImpactResults.cardinalityErrorStd = squeeze(std(cardinalityError, 0, 3));
commImpactResults.runtime = squeeze(mean(runtimes, 3));
commImpactResults.runtimeStd = squeeze(std(runtimes, 0, 3));

commImpactResults.communication.totalDrops = squeeze(mean(totalDrops, 3));
commImpactResults.communication.avgDeliveryRatio = squeeze(mean(avgDeliveryRatio, 3));
commImpactResults.communication.outageCount = squeeze(mean(outageCount, 3));

%% Generate plots
if params.GeneratePlots
    generateCommImpactPlots(commImpactResults, params, actualFusionModes);
end

%% Display summary
if params.Verbose
    fprintf('\n=====================================\n');
    fprintf('Analysis Summary\n');
    fprintf('=====================================\n');
    
    for f = 1:nFusionModes
        fprintf('\n--- %s Fusion Mode ---\n', actualFusionModes{f});
        fprintf('Comm Level | OSPA (mean±std) | Card Error (mean±std) | Runtime (s)\n');
        fprintf('-----------+-----------------+----------------------+------------\n');
        
        for l = 1:nCommLevels
            fprintf('%10d | %7.3f ± %.3f | %9.3f ± %.3f | %8.2f\n', ...
                params.CommLevels(l), ...
                commImpactResults.ospa(f, l), ...
                commImpactResults.ospaStd(f, l), ...
                commImpactResults.cardinalityError(f, l), ...
                commImpactResults.cardinalityErrorStd(f, l), ...
                commImpactResults.runtime(f, l));
        end
    end
    
    fprintf('\nCommunication Statistics:\n');
    fprintf('Comm Level | Total Drops | Delivery Ratio | Outage Count\n');
    fprintf('-----------+-------------+----------------+-------------\n');
    for l = 1:nCommLevels
        fprintf('%10d | %11.1f | %14.3f | %12.1f\n', ...
            params.CommLevels(l), ...
            commImpactResults.communication.totalDrops(l), ...
            commImpactResults.communication.avgDeliveryRatio(l), ...
            commImpactResults.communication.outageCount(l));
    end
    fprintf('\n');
end

%% Save results
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('comm_level_impact_%s.mat', timestamp);
save(filename, 'commImpactResults');

if params.Verbose
    fprintf('Results saved to: %s\n', filename);
end

end

%% Helper function to generate plots
function generateCommImpactPlots(results, params, fusionModes)
    close all;
    
    nCommLevels = length(params.CommLevels);
    nFusionModes = length(fusionModes);
    
    %% Plot 1: OSPA Comparison
    figure('Position', [100, 100, 900, 600]);
    hold on;
    colors = {'b', 'r', 'g', 'm', 'c', 'y'};
    markers = {'o', 's', '^', 'd', 'v', 'p'};
    
    for f = 1:nFusionModes
        colorIdx = mod(f-1, length(colors)) + 1;
        markerIdx = mod(f-1, length(markers)) + 1;
        errorbar(params.CommLevels, results.ospa(f, :), results.ospaStd(f, :), ...
            'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
            'Color', colors{colorIdx}, 'DisplayName', sprintf('%s', fusionModes{f}));
    end
    
    xlabel('Communication Level', 'FontSize', 12);
    ylabel('OSPA Distance', 'FontSize', 12);
    title('OSPA Performance vs Communication Level', 'FontSize', 14);
    grid on;
    legend('Location', 'best');
    set(gca, 'XTick', 0:3);
    xticks = {'0 (Ideal)', '1 (Bandwidth)', '2 (Link Loss)', '3 (Outage)'};
    set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
    
    %% Plot 2: Cardinality Error
    figure('Position', [150, 150, 900, 600]);
    hold on;
    
    for f = 1:nFusionModes
        colorIdx = mod(f-1, length(colors)) + 1;
        markerIdx = mod(f-1, length(markers)) + 1;
        errorbar(params.CommLevels, results.cardinalityError(f, :), ...
            results.cardinalityErrorStd(f, :), ...
            's-', 'LineWidth', 2, 'MarkerSize', 8, ...
            'Color', colors{colorIdx}, 'DisplayName', sprintf('%s', fusionModes{f}));
    end
    
    xlabel('Communication Level', 'FontSize', 12);
    ylabel('Cardinality Error', 'FontSize', 12);
    title('Cardinality Error vs Communication Level', 'FontSize', 14);
    grid on;
    legend('Location', 'best');
    set(gca, 'XTick', 0:3);
    set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
    
    %% Plot 3: Runtime
    figure('Position', [200, 200, 900, 600]);
    hold on;
    
    for f = 1:nFusionModes
        colorIdx = mod(f-1, length(colors)) + 1;
        markerIdx = mod(f-1, length(markers)) + 1;
        errorbar(params.CommLevels, results.runtime(f, :), ...
            results.runtimeStd(f, :), ...
            '^-', 'LineWidth', 2, 'MarkerSize', 8, ...
            'Color', colors{colorIdx}, 'DisplayName', sprintf('%s', fusionModes{f}));
    end
    
    xlabel('Communication Level', 'FontSize', 12);
    ylabel('Runtime (seconds)', 'FontSize', 12);
    title('Runtime vs Communication Level', 'FontSize', 14);
    grid on;
    legend('Location', 'best');
    set(gca, 'XTick', 0:3);
    set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
    
    %% Plot 4: Communication Statistics
    figure('Position', [250, 250, 900, 600]);
    subplot(2, 1, 1);
    bar(results.communication.totalDrops);
    xlabel('Communication Level', 'FontSize', 12);
    ylabel('Total Dropped Measurements', 'FontSize', 12);
    title('Communication Impact: Measurement Drops', 'FontSize', 14);
    grid on;
    set(gca, 'XTick', 1:nCommLevels);
    set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
    
    subplot(2, 1, 2);
    bar(results.communication.avgDeliveryRatio);
    xlabel('Communication Level', 'FontSize', 12);
    ylabel('Delivery Ratio', 'FontSize', 12);
    title('Communication Impact: Delivery Ratio', 'FontSize', 14);
    grid on;
    set(gca, 'XTick', 1:nCommLevels);
    set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
    
    %% Plot 5: Combined Performance Heatmap
    if nFusionModes <= 4
        figure('Position', [300, 300, 1200, 400]);
        for f = 1:nFusionModes
            subplot(1, nFusionModes, f);
            data = results.ospa(f, :);
            bar(data);
            title(sprintf('%s', fusionModes{f}), 'FontSize', 12);
            xlabel('Communication Level', 'FontSize', 10);
            ylabel('OSPA Distance', 'FontSize', 10);
            grid on;
            set(gca, 'XTick', 1:nCommLevels);
            set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
        end
    else
        figure('Position', [300, 300, 1200, 800]);
        subplot(2, ceil(nFusionModes/2), 1);
        data = results.ospa(1:min(2, nFusionModes), :);
        bar(data');
        title('Fusion Modes: GA', 'FontSize', 12);
        xlabel('Communication Level', 'FontSize', 10);
        ylabel('OSPA Distance', 'FontSize', 10);
        grid on;
        set(gca, 'XTick', 1:nCommLevels);
        set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
        legend(fusionModes(1:min(2, nFusionModes)), 'Location', 'best');
        
        subplot(2, ceil(nFusionModes/2), 2);
        data = results.ospa(min(3, nFusionModes):min(4, nFusionModes), :);
        bar(data');
        title('Fusion Modes: AA', 'FontSize', 12);
        xlabel('Communication Level', 'FontSize', 10);
        ylabel('OSPA Distance', 'FontSize', 10);
        grid on;
        set(gca, 'XTick', 1:nCommLevels);
        set(gca, 'XTickLabel', xticks, 'XTickLabelRotation', 15);
        legend(fusionModes(min(3, nFusionModes):min(4, nFusionModes)), 'Location', 'best');
    end
    
end

%% Helper function to safely get struct fields
function val = getField(struct, field, default)
    if isfield(struct, field)
        val = struct.(field);
    else
        val = default;
    end
end
