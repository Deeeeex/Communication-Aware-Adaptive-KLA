function performanceReport = runAdvancedPerformanceAnalysis(varargin)
%RUNADVANCEDPERFORMANCEANALYSIS - Comprehensive performance analysis suite
%   performanceReport = runAdvancedPerformanceAnalysis() runs a complete
%   performance analysis of all implemented filters and algorithms with
%   default parameters.
%
%   performanceReport = runAdvancedPerformanceAnalysis(Name, Value) allows
%   specification of analysis parameters using name-value pairs.
%
%   Name-Value Parameters:
%       'ClutterRates'          - Array of clutter rates to test [default: [5, 10, 15, 20]]
%       'DetectionProbabilities'- Array of detection probs to test [default: [0.7, 0.8, 0.9, 0.95]]
%       'NumberOfTargets'       - Array of target counts [default: [5, 10, 15, 20]]
%       'Algorithms'            - Cell array of algorithms [default: {'LBP', 'Gibbs', 'Murty'}]
%       'MultiSensorModes'      - Cell array of fusion modes [default: {'IC', 'PU', 'GA', 'AA'}]
%       'NumberOfTrials'        - Number of Monte Carlo trials [default: 20]
%       'SaveResults'           - Save results to file [default: true]
%       'GeneratePlots'         - Generate analysis plots [default: true]
%       'Verbose'               - Display progress [default: true]
%       'CommunicationConfig'   - Communication configuration struct [default: struct('level',0)]
%
%   Output:
%       performanceReport - Struct containing comprehensive analysis results
%
%   Examples:
%       % Quick analysis with default parameters
%       report = runAdvancedPerformanceAnalysis();
%
%       % Custom analysis for specific scenarios
%       report = runAdvancedPerformanceAnalysis('ClutterRates', [10, 20], ...
%                                              'NumberOfTrials', 50, ...
%                                              'Algorithms', {'LBP', 'Gibbs'});
%
%   See also runFilters, runMultisensorFilters, computeSimulationOspa

%% Input parsing and validation
defaultClutterRates = [5, 10, 15, 20];
defaultDetectionProbs = [0.7, 0.8, 0.9, 0.95];
defaultNumberOfTargets = [5, 10, 15, 20];
defaultAlgorithms = {'LBP', 'Gibbs', 'Murty'};
% defaultAlgorithms = {'LBP', 'Gibbs'};
defaultMultiSensorModes = {'IC', 'PU', 'GA', 'AA'};
defaultNumberOfTrials = 20;
defaultSaveResults = true;
defaultGeneratePlots = true;
defaultVerbose = true;
defaultCommunicationConfig = struct('level', 0);

p = inputParser;
addParameter(p, 'ClutterRates', defaultClutterRates, @(x) isnumeric(x) && all(x > 0));
addParameter(p, 'DetectionProbabilities', defaultDetectionProbs, @(x) isnumeric(x) && all(x > 0) && all(x <= 1));
addParameter(p, 'NumberOfTargets', defaultNumberOfTargets, @(x) isnumeric(x) && all(x > 0));
addParameter(p, 'Algorithms', defaultAlgorithms, @iscell);
addParameter(p, 'MultiSensorModes', defaultMultiSensorModes, @iscell);
addParameter(p, 'NumberOfTrials', defaultNumberOfTrials, @(x) isnumeric(x) && x > 0);
addParameter(p, 'SaveResults', defaultSaveResults, @islogical);
addParameter(p, 'GeneratePlots', defaultGeneratePlots, @islogical);
addParameter(p, 'Verbose', defaultVerbose, @islogical);
addParameter(p, 'CommunicationConfig', defaultCommunicationConfig, @isstruct);

parse(p, varargin{:});
params = p.Results;

if params.Verbose
    fprintf('Starting Advanced Performance Analysis\n');
    fprintf('=====================================\n');
    fprintf('Parameters:\n');
    fprintf('  Clutter rates: [%s]\n', num2str(params.ClutterRates));
    fprintf('  Detection probabilities: [%s]\n', num2str(params.DetectionProbabilities));
    fprintf('  Number of targets: [%s]\n', num2str(params.NumberOfTargets));
    fprintf('  Algorithms: %s\n', strjoin(params.Algorithms, ', '));
    fprintf('  Multi-sensor modes: %s\n', strjoin(params.MultiSensorModes, ', '));
    fprintf('  Number of trials: %d\n', params.NumberOfTrials);
    fprintf('\n');
end

%% Initialize results structure
performanceReport = struct();
performanceReport.metadata = struct();
performanceReport.metadata.timestamp = datetime('now');
performanceReport.metadata.parameters = params;
performanceReport.metadata.matlabVersion = version;

%% Single-sensor analysis
if params.Verbose
    fprintf('Running single-sensor analysis...\n');
end

singleSensorResults = runSingleSensorAnalysis(params);
performanceReport.singleSensor = singleSensorResults;

%% Multi-sensor analysis
if params.Verbose
    fprintf('Running multi-sensor analysis...\n');
end

multiSensorResults = runMultiSensorAnalysis(params);
performanceReport.multiSensor = multiSensorResults;

%% Algorithm comparison
if params.Verbose
    fprintf('Performing algorithm comparison...\n');
end

algorithmComparison = runAlgorithmComparison(params);
performanceReport.algorithmComparison = algorithmComparison;

%% Scalability analysis
if params.Verbose
    fprintf('Analyzing scalability...\n');
end

scalabilityResults = runScalabilityAnalysis(params);
performanceReport.scalability = scalabilityResults;

%% Generate comprehensive plots
if params.GeneratePlots
    if params.Verbose
        fprintf('Generating analysis plots...\n');
    end
    generatePerformancePlots(performanceReport);
end

%% Save results
if params.SaveResults
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('performance_analysis_%s.mat', timestamp);
    save(filename, 'performanceReport');
    if params.Verbose
        fprintf('Results saved to: %s\n', filename);
    end
end

if params.Verbose
    fprintf('Analysis complete!\n');
end

end

%% Single-sensor analysis function
function results = runSingleSensorAnalysis(params)
    results = struct();
    
    nClutter = length(params.ClutterRates);
    nDetection = length(params.DetectionProbabilities);
    nTargets = length(params.NumberOfTargets);
    nAlgorithms = length(params.Algorithms);
    nTrials = params.NumberOfTrials;
    
    % --- 修改点：创建临时矩阵，不直接操作 results 结构体 ---
    ospa_clutter = zeros(nAlgorithms, nClutter, nTrials);
    runtime_clutter = zeros(nAlgorithms, nClutter, nTrials);
    
    ospa_detection = zeros(nAlgorithms, nDetection, nTrials);
    runtime_detection = zeros(nAlgorithms, nDetection, nTrials);
    
    ospa_targets = zeros(nAlgorithms, nTargets, nTrials);
    runtime_targets = zeros(nAlgorithms, nTargets, nTrials);
    % ---------------------------------------------------

    % Clutter rate analysis
    for c = 1:nClutter
        for a = 1:nAlgorithms
            % 提取当前循环需要的参数，避免在 parfor 中广播整个 params
            currentClutter = params.ClutterRates(c);
            currentAlg = params.Algorithms{a};
            
            parfor trial = 1:nTrials
                model = generateModel(currentClutter, 0.9, currentAlg, 'Fixed');
                [groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model);
                
                tic;
                stateEstimates = runLmbFilter(model, measurements);
                runtime = toc;
                
                ospaScores = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
                
                % 赋值给临时矩阵
                ospa_clutter(a, c, trial) = mean(ospaScores);
                runtime_clutter(a, c, trial) = runtime;
            end
        end
    end
    
    % Detection probability analysis (同理修改)
    for d = 1:nDetection
        for a = 1:nAlgorithms
            currentDet = params.DetectionProbabilities(d);
            currentAlg = params.Algorithms{a};
            parfor trial = 1:nTrials
                model = generateModel(10, currentDet, currentAlg, 'Fixed');
                [groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model);
                tic;
                stateEstimates = runLmbFilter(model, measurements);
                runtime = toc;
                ospaScores = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
                
                ospa_detection(a, d, trial) = mean(ospaScores);
                runtime_detection(a, d, trial) = runtime;
            end
        end
    end

    % Target count analysis (同理修改)
    for t = 1:nTargets
        for a = 1:nAlgorithms
            currentTgt = params.NumberOfTargets(t);
            currentAlg = params.Algorithms{a};
            parfor trial = 1:nTrials
                model = generateModel(10, 0.9, currentAlg, 'Random', currentTgt);
                [groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model, currentTgt);
                tic;
                stateEstimates = runLmbFilter(model, measurements);
                runtime = toc;
                ospaScores = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
                
                ospa_targets(a, t, trial) = mean(ospaScores);
                runtime_targets(a, t, trial) = runtime;
            end
        end
    end
    
    % --- 循环结束后统一赋值回结果结构体 ---
    results.ospa.clutter = ospa_clutter;
    results.runtime.clutter = runtime_clutter;
    results.ospa.detection = ospa_detection;
    results.runtime.detection = runtime_detection;
    results.ospa.targets = ospa_targets;
    results.runtime.targets = runtime_targets;
    
    results.clutterRates = params.ClutterRates;
    results.detectionProbabilities = params.DetectionProbabilities;
    results.numberOfTargets = params.NumberOfTargets;
    results.algorithms = params.Algorithms;
end

%% Multi-sensor analysis function
function results = runMultiSensorAnalysis(params)
    results = struct();
    
    nModes = length(params.MultiSensorModes);
    nTrials = params.NumberOfTrials;
    commConfig = params.CommunicationConfig;
    if ~isstruct(commConfig)
        commConfig = struct('level', 0);
    end
    
    % Sensor configurations to test
    sensorConfigs = {
        struct('nSensors', 2, 'clutterRates', [5, 8], 'detectionProbs', [0.9, 0.85], 'precision', [4, 3]),
        struct('nSensors', 3, 'clutterRates', [5, 8, 10], 'detectionProbs', [0.9, 0.85, 0.8], 'precision', [4, 3, 2]),
        struct('nSensors', 4, 'clutterRates', [5, 8, 10, 12], 'detectionProbs', [0.9, 0.85, 0.8, 0.75], 'precision', [4, 3, 2, 1])
    };
    
    nConfigs = length(sensorConfigs);
    
    % 创建临时矩阵
    temp_ospa = zeros(nModes, nConfigs, nTrials);
    temp_runtime = zeros(nModes, nConfigs, nTrials);
    temp_cardError = zeros(nModes, nConfigs, nTrials);
    
    for config = 1:nConfigs
        sc = sensorConfigs{config};
        for mode = 1:nModes
            currentMode = params.MultiSensorModes{mode};
            commConfigLocal = commConfig;
            parfor trial = 1:nTrials
                model = generateMultisensorModel(sc.nSensors, sc.clutterRates, ...
                sc.detectionProbs, sc.precision, params.MultiSensorModes{mode}, 'LBP', 'Fixed');
            
            [groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
            [measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfigLocal);
            
            tic;
            switch params.MultiSensorModes{mode}
                case 'IC'
                    stateEstimates = runIcLmbFilter(model, measurementsDelivered);
                case {'PU', 'GA', 'AA'}
                    model.lmbParallelUpdateMode = params.MultiSensorModes{mode};
                    stateEstimates = runParallelUpdateLmbFilter(model, measurementsDelivered, commStats);
            end
            runtime = toc;
            
            ospaScores = computeSimulationOspa(model, groundTruthRfs, stateEstimates);
            % cardinalityErrors = computeCardinalityErrors(groundTruthRfs, stateEstimates);
                % 赋值给临时变量
                temp_ospa(mode, config, trial) = mean(ospaScores);
                temp_runtime(mode, config, trial) = runtime;
                % temp_cardError(mode, config, trial) = mean(cardinalityErrors);
            end
        end
    end
    
    % 赋值回结构体
    results.ospa = temp_ospa;
    results.runtime = temp_runtime;
    % results.cardinalityError = temp_cardError;
    results.sensorConfigs = sensorConfigs;
    results.modes = params.MultiSensorModes;
end

%% Algorithm comparison function
function results = runAlgorithmComparison(params)
    results = struct();
    
    % Standard scenario for comparison
    model = generateModel(10, 0.9, 'LBP', 'Fixed');
    [groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model);
    
    nAlgorithms = length(params.Algorithms);
    nTrials = params.NumberOfTrials;
    
    % 创建临时矩阵
    temp_ospa = zeros(nAlgorithms, nTrials);
    temp_runtime = zeros(nAlgorithms, nTrials);
    temp_convergence = zeros(nAlgorithms, nTrials);
    
    for a = 1:nAlgorithms
        algorithm = params.Algorithms{a};
        
        parfor trial = 1:nTrials
            testModel = model;
            testModel.dataAssociationMethod = algorithm;
            
            % Algorithm-specific parameters
            switch algorithm
                case 'Gibbs'
                    testModel.numberOfSamples = 5000;
                case 'Murty'
                    testModel.numberOfAssignments = 100;
                case 'LBP'
                    testModel.maximumNumberOfLbpIterations = 1000;
            end
            
            tic;
            stateEstimates = runLmbFilter(testModel, measurements);
            runtime = toc;
            
            ospaScores = computeSimulationOspa(testModel, groundTruthRfs, stateEstimates);
            
            temp_ospa(a, trial) = mean(ospaScores);
            temp_runtime(a, trial) = runtime;
            temp_convergence(a, trial) = 1;
            
            % Measure convergence (algorithm-specific)
            % if strcmp(algorithm, 'LBP')
            %     results.convergence(a, trial) = 1; % Assume convergence for now
            % else
            %     results.convergence(a, trial) = 1;
            % end
        end
    end
    
    % 赋值回结构体
    results.ospa = temp_ospa;
    results.runtime = temp_runtime;
    results.convergence = temp_convergence;
    results.algorithms = params.Algorithms;
end

%% Scalability analysis function
function results = runScalabilityAnalysis(params)
    results = struct();
    
    % Test scaling with number of objects and measurements
    objectCounts = [5, 10, 20, 30, 40, 50];
    clutterCounts = [5, 10, 20, 30, 40, 50];
    
    nObjects = length(objectCounts);
    nClutter = length(clutterCounts);
    nAlgorithms = length(params.Algorithms);
    
    results.runtime.objects = zeros(nAlgorithms, nObjects);
    results.runtime.clutter = zeros(nAlgorithms, nClutter);
    results.memory.objects = zeros(nAlgorithms, nObjects);
    results.memory.clutter = zeros(nAlgorithms, nClutter);
    
    % Object count scaling
    for o = 1:nObjects
        for a = 1:nAlgorithms
            model = generateModel(10, 0.9, params.Algorithms{a}, 'Random', objectCounts(o));
            [~, measurements] = generateGroundTruth(model, objectCounts(o));
            
            % Measure memory before
            memBefore = getMemoryUsage();
            
            tic;
            stateEstimates = runLmbFilter(model, measurements);
            runtime = toc;
            
            % Measure memory after
            memAfter = getMemoryUsage();
            
            results.runtime.objects(a, o) = runtime;
            results.memory.objects(a, o) = memAfter - memBefore;
        end
    end
    
    % Clutter count scaling
    for c = 1:nClutter
        for a = 1:nAlgorithms
            model = generateModel(clutterCounts(c), 0.9, params.Algorithms{a}, 'Fixed');
            [~, measurements] = generateGroundTruth(model);
            
            % Measure memory before
            memBefore = getMemoryUsage();
            
            tic;
            stateEstimates = runLmbFilter(model, measurements);
            runtime = toc;
            
            % Measure memory after
            memAfter = getMemoryUsage();
            
            results.runtime.clutter(a, c) = runtime;
            results.memory.clutter(a, c) = memAfter - memBefore;
        end
    end
    
    results.objectCounts = objectCounts;
    results.clutterCounts = clutterCounts;
    results.algorithms = params.Algorithms;
end

%% Helper function to compute cardinality errors
function errors = computeCardinalityErrors(groundTruth, stateEstimates)
    T = length(groundTruth.labels);
    errors = zeros(T, 1);
    
    for t = 1:T
        trueCardinality = size(groundTruth.labels{t}, 2);
        estimatedCardinality = size(stateEstimates.labels{t}, 2);
        errors(t) = abs(trueCardinality - estimatedCardinality);
    end
end

%% Helper function to get memory usage (simplified)
function memUsage = getMemoryUsage()
    % This is a simplified implementation
    % In practice, you might use more sophisticated memory monitoring
    s = whos;
    memUsage = sum([s.bytes]) / 1024^2; % Convert to MB
end

%% Plot generation function
function generatePerformancePlots(performanceReport)
    % Create comprehensive visualization of results
    
    % Single-sensor performance plots
    figure('Name', 'Single-Sensor Performance Analysis', 'Position', [100, 100, 1200, 800]);
    
    % OSPA vs Clutter Rate
    subplot(2, 3, 1);
    singleSensor = performanceReport.singleSensor;
    meanOspa = squeeze(mean(singleSensor.ospa.clutter, 3));
    stdOspa = squeeze(std(singleSensor.ospa.clutter, 0, 3));
    
    hold on;
    colors = {'b', 'r', 'g', 'm'};
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.clutterRates, meanOspa(a, :), stdOspa(a, :), ...
            'Color', colors{a}, 'LineWidth', 2, 'DisplayName', singleSensor.algorithms{a});
    end
    xlabel('Clutter Rate');
    ylabel('Mean OSPA Distance');
    title('Performance vs Clutter Rate');
    legend('show');
    grid on;
    
    % Runtime vs Clutter Rate
    subplot(2, 3, 2);
    meanRuntime = squeeze(mean(singleSensor.runtime.clutter, 3));
    stdRuntime = squeeze(std(singleSensor.runtime.clutter, 0, 3));
    
    hold on;
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.clutterRates, meanRuntime(a, :), stdRuntime(a, :), ...
            'Color', colors{a}, 'LineWidth', 2, 'DisplayName', singleSensor.algorithms{a});
    end
    xlabel('Clutter Rate');
    ylabel('Runtime (seconds)');
    title('Runtime vs Clutter Rate');
    legend('show');
    grid on;
    
    % Similar plots for other parameters...
    % (Additional subplot code would go here)
    
    % Multi-sensor comparison
    figure('Name', 'Multi-Sensor Performance Comparison', 'Position', [200, 200, 1000, 600]);
    
    multiSensor = performanceReport.multiSensor;
    meanMultiOspa = squeeze(mean(multiSensor.ospa, 3));
    
    subplot(1, 2, 1);
    bar(meanMultiOspa);
    xlabel('Sensor Configuration');
    ylabel('Mean OSPA Distance');
    title('Multi-Sensor OSPA Performance');
    legend(multiSensor.modes, 'Location', 'best');
    grid on;
    
    subplot(1, 2, 2);
    meanMultiRuntime = squeeze(mean(multiSensor.runtime, 3));
    bar(meanMultiRuntime);
    xlabel('Sensor Configuration');
    ylabel('Runtime (seconds)');
    title('Multi-Sensor Runtime Performance');
    legend(multiSensor.modes, 'Location', 'best');
    grid on;
    
    % Scalability plots
    figure('Name', 'Scalability Analysis', 'Position', [300, 300, 1000, 600]);
    
    scalability = performanceReport.scalability;
    
    subplot(1, 2, 1);
    loglog(scalability.objectCounts, scalability.runtime.objects', 'o-', 'LineWidth', 2);
    xlabel('Number of Objects');
    ylabel('Runtime (seconds)');
    title('Runtime Scalability vs Object Count');
    legend(scalability.algorithms, 'Location', 'best');
    grid on;
    
    subplot(1, 2, 2);
    loglog(scalability.clutterCounts, scalability.runtime.clutter', 'o-', 'LineWidth', 2);
    xlabel('Clutter Rate');
    ylabel('Runtime (seconds)');
    title('Runtime Scalability vs Clutter Rate');
    legend(scalability.algorithms, 'Location', 'best');
    grid on;
end
