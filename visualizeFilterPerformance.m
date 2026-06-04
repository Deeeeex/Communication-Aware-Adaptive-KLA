function visualizeFilterPerformance(varargin)
%VISUALIZEFILTERPERFORMANCE - Advanced visualization of filter performance with mathematical annotations
%   visualizeFilterPerformance() creates comprehensive performance plots
%   with default parameters.
%
%   visualizeFilterPerformance(Name, Value) allows customization using
%   name-value pairs.
%
%   Mathematical Framework:
%   The visualizations are based on the OSPA (Optimal Sub-Pattern Assignment) metric:
%
%   $$\text{OSPA}_p^{(c)}(X, Y) = \left(\frac{1}{n}\left[\sum_{i=1}^m d^{(c)}(x_i, y_{\sigma(i)})^p + c^p(n-m)\right]\right)^{1/p}$$
%
%   where:
%   - $X = \{x_1, \ldots, x_m\}$ and $Y = \{y_1, \ldots, y_n\}$ are finite sets
%   - $d^{(c)}(x,y) = \min(d(x,y), c)$ is the cut-off distance
%   - $\sigma$ is the optimal assignment
%   - $c$ is the cut-off parameter
%   - $p$ is the order parameter
%
%   Name-Value Parameters:
%       'DataFile'              - MAT file containing results (default: auto-detect latest)
%       'PlotTypes'             - Cell array of plot types (default: {'OSPA', 'Runtime', 'Cardinality'})
%       'Algorithms'            - Algorithms to compare (default: all available)
%       'SavePlots'             - Save plots to files (default: true)
%       'PlotFormat'            - Output format: 'png', 'pdf', 'eps' (default: 'png')
%       'ShowMath'              - Display mathematical annotations (default: true)
%       'FontSize'              - Font size for plots (default: 12)
%       'LineWidth'             - Line width for plots (default: 2)
%       'MarkerSize'            - Marker size (default: 8)
%       'ColorScheme'           - Color scheme: 'default', 'colorblind', 'grayscale' (default: 'default')
%
%   Examples:
%       % Basic visualization
%       visualizeFilterPerformance();
%
%       % Custom visualization
%       visualizeFilterPerformance('PlotTypes', {'OSPA', 'Runtime'}, ...
%                                 'ColorScheme', 'colorblind', ...
%                                 'SavePlots', true);
%
%   See also runAdvancedPerformanceAnalysis, computeSimulationOspa

%% Input parsing
p = inputParser;
addParameter(p, 'DataFile', '', @ischar);
addParameter(p, 'PlotTypes', {'OSPA', 'Runtime', 'Cardinality'}, @iscell);
addParameter(p, 'Algorithms', {}, @iscell);
addParameter(p, 'SavePlots', true, @islogical);
addParameter(p, 'PlotFormat', 'png', @(x) ismember(x, {'png', 'pdf', 'eps', 'fig'}));
addParameter(p, 'ShowMath', true, @islogical);
addParameter(p, 'FontSize', 12, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LineWidth', 2, @(x) isnumeric(x) && x > 0);
addParameter(p, 'MarkerSize', 8, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ColorScheme', 'default', @(x) ismember(x, {'default', 'colorblind', 'grayscale'}));

parse(p, varargin{:});
params = p.Results;

%% Load data
if isempty(params.DataFile)
    % Auto-detect latest performance analysis file
    files = dir('performance_analysis_*.mat');
    if isempty(files)
        error('No performance analysis files found. Please run runAdvancedPerformanceAnalysis first.');
    end
    [~, idx] = max([files.datenum]);
    dataFile = files(idx).name;
    fprintf('Loading latest performance data: %s\n', dataFile);
else
    dataFile = params.DataFile;
end

data = load(dataFile);
performanceReport = data.performanceReport;

%% Set up color scheme
colors = getColorScheme(params.ColorScheme);

%% Set default font properties
set(0, 'DefaultAxesFontSize', params.FontSize);
set(0, 'DefaultTextFontSize', params.FontSize);
set(0, 'DefaultLineLineWidth', params.LineWidth);
set(0, 'DefaultLineMarkerSize', params.MarkerSize);

%% Generate plots based on requested types
for i = 1:length(params.PlotTypes)
    plotType = params.PlotTypes{i};
    
    switch plotType
        case 'OSPA'
            createOSPAPlots(performanceReport, colors, params);
        case 'Runtime'
            createRuntimePlots(performanceReport, colors, params);
        case 'Cardinality'
            createCardinalityPlots(performanceReport, colors, params);
        case 'Scalability'
            createScalabilityPlots(performanceReport, colors, params);
        case 'Comparison'
            createComparisonPlots(performanceReport, colors, params);
        otherwise
            warning('Unknown plot type: %s', plotType);
    end
end

fprintf('Visualization complete!\n');

end

%% OSPA Performance Plots
function createOSPAPlots(performanceReport, colors, params)
    if ~isfield(performanceReport, 'singleSensor')
        warning('No single-sensor data available for OSPA plots');
        return;
    end
    
    singleSensor = performanceReport.singleSensor;
    
    % Create main OSPA figure
    fig = figure('Name', 'OSPA Performance Analysis', 'Position', [100, 100, 1400, 900]);
    
    % Plot 1: OSPA vs Clutter Rate
    subplot(2, 3, 1);
    meanOspa = squeeze(mean(singleSensor.ospa.clutter, 3));
    stdOspa = squeeze(std(singleSensor.ospa.clutter, 0, 3));
    
    hold on;
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.clutterRates, meanOspa(a, :), stdOspa(a, :), ...
            'Color', colors{mod(a-1, length(colors))+1}, 'LineWidth', params.LineWidth, ...
            'DisplayName', singleSensor.algorithms{a}, 'Marker', 'o', 'MarkerSize', params.MarkerSize);
    end
    
    xlabel('Clutter Rate $\lambda_c$ (returns/scan)', 'Interpreter', 'latex');
    ylabel('Mean OSPA Distance', 'Interpreter', 'latex');
    title('OSPA Performance vs Clutter Density', 'Interpreter', 'latex');
    legend('show', 'Location', 'best');
    grid on;
    
    if params.ShowMath
        % Add mathematical annotation
        text(0.02, 0.98, '$\text{OSPA}_2^{(10)}(X, Y)$', 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'Interpreter', 'latex', 'FontSize', params.FontSize+2, ...
            'BackgroundColor', 'white', 'EdgeColor', 'black');
    end
    
    % Plot 2: OSPA vs Detection Probability
    subplot(2, 3, 2);
    meanOspaDetection = squeeze(mean(singleSensor.ospa.detection, 3));
    stdOspaDetection = squeeze(std(singleSensor.ospa.detection, 0, 3));
    
    hold on;
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.detectionProbabilities, meanOspaDetection(a, :), stdOspaDetection(a, :), ...
            'Color', colors{mod(a-1, length(colors))+1}, 'LineWidth', params.LineWidth, ...
            'DisplayName', singleSensor.algorithms{a}, 'Marker', 's', 'MarkerSize', params.MarkerSize);
    end
    
    xlabel('Detection Probability $P_D$', 'Interpreter', 'latex');
    ylabel('Mean OSPA Distance', 'Interpreter', 'latex');
    title('OSPA Performance vs Detection Probability', 'Interpreter', 'latex');
    legend('show', 'Location', 'best');
    grid on;
    
    % Plot 3: OSPA vs Number of Targets
    subplot(2, 3, 3);
    meanOspaTargets = squeeze(mean(singleSensor.ospa.targets, 3));
    stdOspaTargets = squeeze(std(singleSensor.ospa.targets, 0, 3));
    
    hold on;
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.numberOfTargets, meanOspaTargets(a, :), stdOspaTargets(a, :), ...
            'Color', colors{mod(a-1, length(colors))+1}, 'LineWidth', params.LineWidth, ...
            'DisplayName', singleSensor.algorithms{a}, 'Marker', '^', 'MarkerSize', params.MarkerSize);
    end
    
    xlabel('Number of Targets $N$', 'Interpreter', 'latex');
    ylabel('Mean OSPA Distance', 'Interpreter', 'latex');
    title('OSPA Performance vs Target Density', 'Interpreter', 'latex');
    legend('show', 'Location', 'best');
    grid on;
    
    % Plot 4: OSPA Distribution (Box Plot)
    subplot(2, 3, 4);
    ospaData = [];
    ospaLabels = {};
    for a = 1:length(singleSensor.algorithms)
        ospaData = [ospaData; squeeze(singleSensor.ospa.clutter(a, :, :))'];
        ospaLabels = [ospaLabels; repmat(singleSensor.algorithms(a), size(singleSensor.ospa.clutter, 2), 1)];
    end
    
    boxplot(ospaData(:), ospaLabels, 'Colors', cell2mat(colors(1:length(singleSensor.algorithms))'));
    xlabel('Algorithm', 'Interpreter', 'latex');
    ylabel('OSPA Distance Distribution', 'Interpreter', 'latex');
    title('OSPA Distribution by Algorithm', 'Interpreter', 'latex');
    grid on;
    
    % Plot 5: Statistical Significance Test Results
    subplot(2, 3, 5);
    [pValues, testStats] = performStatisticalTests(singleSensor);
    
    imagesc(pValues);
    colorbar;
    colormap(gray);
    
    set(gca, 'XTick', 1:length(singleSensor.algorithms));
    set(gca, 'YTick', 1:length(singleSensor.algorithms));
    set(gca, 'XTickLabel', singleSensor.algorithms);
    set(gca, 'YTickLabel', singleSensor.algorithms);
    xlabel('Algorithm', 'Interpreter', 'latex');
    ylabel('Algorithm', 'Interpreter', 'latex');
    title('Statistical Significance ($p$-values)', 'Interpreter', 'latex');
    
    % Add p-value annotations
    for i = 1:length(singleSensor.algorithms)
        for j = 1:length(singleSensor.algorithms)
            if i ~= j
                text(j, i, sprintf('%.3f', pValues(i,j)), 'HorizontalAlignment', 'center', ...
                    'Color', pValues(i,j) < 0.05 ? 'red' : 'black', 'FontWeight', 'bold');
            end
        end
    end
    
    % Plot 6: Performance Summary
    subplot(2, 3, 6);
    % Radar plot of normalized performance metrics
    createRadarPlot(singleSensor, colors, params);
    
    % Save figure
    if params.SavePlots
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        filename = sprintf('ospa_analysis_%s.%s', timestamp, params.PlotFormat);
        savePlot(fig, filename, params.PlotFormat);
    end
end

%% Runtime Performance Plots
function createRuntimePlots(performanceReport, colors, params)
    if ~isfield(performanceReport, 'singleSensor')
        warning('No single-sensor data available for runtime plots');
        return;
    end
    
    singleSensor = performanceReport.singleSensor;
    
    fig = figure('Name', 'Runtime Performance Analysis', 'Position', [200, 200, 1200, 800]);
    
    % Plot 1: Runtime vs Clutter Rate
    subplot(2, 2, 1);
    meanRuntime = squeeze(mean(singleSensor.runtime.clutter, 3));
    stdRuntime = squeeze(std(singleSensor.runtime.clutter, 0, 3));
    
    hold on;
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.clutterRates, meanRuntime(a, :), stdRuntime(a, :), ...
            'Color', colors{mod(a-1, length(colors))+1}, 'LineWidth', params.LineWidth, ...
            'DisplayName', singleSensor.algorithms{a}, 'Marker', 'o', 'MarkerSize', params.MarkerSize);
    end
    
    xlabel('Clutter Rate $\lambda_c$', 'Interpreter', 'latex');
    ylabel('Runtime (seconds)', 'Interpreter', 'latex');
    title('Runtime vs Clutter Density', 'Interpreter', 'latex');
    legend('show', 'Location', 'best');
    grid on;
    set(gca, 'YScale', 'log');
    
    % Plot 2: Runtime vs Number of Targets
    subplot(2, 2, 2);
    meanRuntimeTargets = squeeze(mean(singleSensor.runtime.targets, 3));
    stdRuntimeTargets = squeeze(std(singleSensor.runtime.targets, 0, 3));
    
    hold on;
    for a = 1:length(singleSensor.algorithms)
        errorbar(singleSensor.numberOfTargets, meanRuntimeTargets(a, :), stdRuntimeTargets(a, :), ...
            'Color', colors{mod(a-1, length(colors))+1}, 'LineWidth', params.LineWidth, ...
            'DisplayName', singleSensor.algorithms{a}, 'Marker', 's', 'MarkerSize', params.MarkerSize);
    end
    
    xlabel('Number of Targets $N$', 'Interpreter', 'latex');
    ylabel('Runtime (seconds)', 'Interpreter', 'latex');
    title('Runtime vs Target Density', 'Interpreter', 'latex');
    legend('show', 'Location', 'best');
    grid on;
    set(gca, 'YScale', 'log');
    
    if params.ShowMath
        % Add complexity annotations
        text(0.02, 0.98, 'LBP: $O(NM \cdot I)$', 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'Interpreter', 'latex', 'FontSize', params.FontSize, ...
            'BackgroundColor', 'white', 'EdgeColor', 'black');
        text(0.02, 0.88, 'Gibbs: $O(NM \cdot S)$', 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'Interpreter', 'latex', 'FontSize', params.FontSize, ...
            'BackgroundColor', 'white', 'EdgeColor', 'black');
        text(0.02, 0.78, 'Murty: $O(K \cdot N^3)$', 'Units', 'normalized', ...
            'VerticalAlignment', 'top', 'Interpreter', 'latex', 'FontSize', params.FontSize, ...
            'BackgroundColor', 'white', 'EdgeColor', 'black');
    end
    
    % Plot 3: Runtime Efficiency (OSPA/Runtime)
    subplot(2, 2, 3);
    efficiency = meanOspa ./ meanRuntime;
    
    bar(efficiency');
    xlabel('Algorithm', 'Interpreter', 'latex');
    ylabel('Efficiency (OSPA/Runtime)', 'Interpreter', 'latex');
    title('Algorithm Efficiency Comparison', 'Interpreter', 'latex');
    set(gca, 'XTickLabel', singleSensor.algorithms);
    grid on;
    
    % Plot 4: Runtime Distribution
    subplot(2, 2, 4);
    runtimeData = [];
    runtimeLabels = {};
    for a = 1:length(singleSensor.algorithms)
        runtimeData = [runtimeData; squeeze(singleSensor.runtime.clutter(a, :, :))'];
        runtimeLabels = [runtimeLabels; repmat(singleSensor.algorithms(a), size(singleSensor.runtime.clutter, 2), 1)];
    end
    
    boxplot(runtimeData(:), runtimeLabels, 'Colors', cell2mat(colors(1:length(singleSensor.algorithms))'));
    xlabel('Algorithm', 'Interpreter', 'latex');
    ylabel('Runtime Distribution (seconds)', 'Interpreter', 'latex');
    title('Runtime Distribution by Algorithm', 'Interpreter', 'latex');
    set(gca, 'YScale', 'log');
    grid on;
    
    % Save figure
    if params.SavePlots
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        filename = sprintf('runtime_analysis_%s.%s', timestamp, params.PlotFormat);
        savePlot(fig, filename, params.PlotFormat);
    end
end

%% Helper function to get color scheme
function colors = getColorScheme(scheme)
    switch scheme
        case 'default'
            colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'};
        case 'colorblind'
            colors = {'#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'};
        case 'grayscale'
            colors = {'#000000', '#404040', '#808080', '#c0c0c0', '#606060', '#a0a0a0'};
        otherwise
            colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'};
    end
end

%% Statistical significance testing
function [pValues, testStats] = performStatisticalTests(singleSensor)
    nAlgorithms = length(singleSensor.algorithms);
    pValues = eye(nAlgorithms);
    testStats = eye(nAlgorithms);
    
    % Use Wilcoxon rank-sum test for pairwise comparisons
    for i = 1:nAlgorithms
        for j = 1:nAlgorithms
            if i ~= j
                data1 = squeeze(singleSensor.ospa.clutter(i, :, :));
                data2 = squeeze(singleSensor.ospa.clutter(j, :, :));
                
                [p, h, stats] = ranksum(data1(:), data2(:));
                pValues(i, j) = p;
                testStats(i, j) = stats.ranksum;
            end
        end
    end
end

%% Radar plot for performance summary
function createRadarPlot(singleSensor, colors, params)
    % Normalize metrics (lower is better, so invert)
    meanOspa = squeeze(mean(singleSensor.ospa.clutter, [2, 3]));
    meanRuntime = squeeze(mean(singleSensor.runtime.clutter, [2, 3]));
    
    % Normalize to [0, 1] and invert (higher is better)
    ospaScore = 1 - (meanOspa - min(meanOspa)) ./ (max(meanOspa) - min(meanOspa));
    runtimeScore = 1 - (meanRuntime - min(meanRuntime)) ./ (max(meanRuntime) - min(meanRuntime));
    
    % Create simple bar chart instead of complex radar plot
    metrics = [ospaScore, runtimeScore];
    bar(metrics);
    
    xlabel('Algorithm', 'Interpreter', 'latex');
    ylabel('Normalized Performance Score', 'Interpreter', 'latex');
    title('Overall Performance Summary', 'Interpreter', 'latex');
    set(gca, 'XTickLabel', singleSensor.algorithms);
    legend({'Accuracy (1-OSPA)', 'Speed (1-Runtime)'}, 'Location', 'best');
    grid on;
end

%% Save plot function
function savePlot(fig, filename, format)
    switch format
        case 'png'
            print(fig, filename, '-dpng', '-r300');
        case 'pdf'
            print(fig, filename, '-dpdf', '-r300');
        case 'eps'
            print(fig, filename, '-depsc', '-r300');
        case 'fig'
            savefig(fig, filename);
    end
    
    fprintf('Plot saved: %s\n', filename);
end

%% Placeholder functions for other plot types
function createCardinalityPlots(performanceReport, colors, params)
    % Implementation for cardinality plots
    fprintf('Cardinality plots not yet implemented.\n');
end

function createScalabilityPlots(performanceReport, colors, params)
    % Implementation for scalability plots
    fprintf('Scalability plots not yet implemented.\n');
end

function createComparisonPlots(performanceReport, colors, params)
    % Implementation for comparison plots
    fprintf('Comparison plots not yet implemented.\n');
end
