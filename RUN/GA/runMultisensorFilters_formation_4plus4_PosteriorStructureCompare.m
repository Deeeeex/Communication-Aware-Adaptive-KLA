function [reportPath, summary] = runMultisensorFilters_formation_4plus4_PosteriorStructureCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, adaptiveFusionOverrides)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_POSTERIORSTRUCTURECOMPARE
% Compare static weak structure prior against posterior-structure-consistency.

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 5;
end
if nargin < 2 || isempty(baseSeed)
    baseSeed = 1;
end
if nargin < 3 || isempty(useFixedSeed)
    useFixedSeed = true;
end
if nargin < 4 || isempty(writeReport)
    writeReport = true;
end
if nargin < 5 || isempty(adaptiveFusionOverrides)
    adaptiveFusionOverrides = struct();
end

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
setPath;

baseOverrides = mergeStructFields(struct( ...
    'usePosteriorStructureConsistency', false), adaptiveFusionOverrides);
experimentOverrides = mergeStructFields(struct( ...
    'usePosteriorStructureConsistency', true), adaptiveFusionOverrides);

[~, baseSummary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, struct(), false, ...
    'structure-aware-decoupled-kla', baseOverrides, 4);
[~, experimentSummary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, struct(), false, ...
    'structure-aware-decoupled-kla', experimentOverrides, 4);

reportPath = '';
summary = buildSummary(baseSummary, experimentSummary, baseOverrides, experimentOverrides);

fprintf('=====================================\n');
fprintf('GA Posterior Structure Comparison (N=%d)\n', numberOfTrials);
fprintf('Static structure prior -> posterior-structure-consistency\n');
fprintf('OSPA consensus error: %.6f -> %.6f\n', summary.consensus.ospaBase, summary.consensus.ospaAdaptive);
fprintf('Matched localization disagreement: %.6f -> %.6f\n', summary.consensus.posBase, summary.consensus.posAdaptive);
fprintf('Cardinality dispersion: %.6f -> %.6f\n', summary.consensus.cardBase, summary.consensus.cardAdaptive);

if writeReport
    reportDir = fullfile(projectRoot, 'RUN', 'GA');
    if ~exist(reportDir, 'dir')
        mkdir(reportDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    reportPath = fullfile(reportDir, sprintf('GA_POSTERIOR_STRUCTURE_COMPARE_%s.md', timestamp));
    writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, summary);
    fprintf('Report written: %s\n', reportPath);
end
end

function summary = buildSummary(baseSummary, experimentSummary, baseOverrides, experimentOverrides)
summary = struct();
summary.consensus.ospaBase = extractScalar(baseSummary.consensus.ospa);
summary.consensus.ospaAdaptive = extractScalar(experimentSummary.consensus.ospa);
summary.consensus.posBase = extractScalar(baseSummary.consensus.pos);
summary.consensus.posAdaptive = extractScalar(experimentSummary.consensus.pos);
summary.consensus.cardBase = extractScalar(baseSummary.consensus.card);
summary.consensus.cardAdaptive = extractScalar(experimentSummary.consensus.card);
summary.local.eOspaBase = reshape(baseSummary.local.eOspa, 1, []);
summary.local.eOspaAdaptive = reshape(experimentSummary.local.eOspa, 1, []);
summary.local.rmseBase = reshape(baseSummary.local.rmse, 1, []);
summary.local.rmseAdaptive = reshape(experimentSummary.local.rmse, 1, []);
summary.delta.ospa = summary.consensus.ospaAdaptive - summary.consensus.ospaBase;
summary.delta.pos = summary.consensus.posAdaptive - summary.consensus.posBase;
summary.delta.card = summary.consensus.cardAdaptive - summary.consensus.cardBase;
summary.delta.localEospa = mean(summary.local.eOspaAdaptive) - mean(summary.local.eOspaBase);
summary.delta.localRmse = mean(summary.local.rmseAdaptive) - mean(summary.local.rmseBase);
summary.baseOverrides = baseOverrides;
summary.experimentOverrides = experimentOverrides;
end

function writeComparisonReport(reportPath, numberOfTrials, baseSeed, useFixedSeed, summary)
fid = fopen(reportPath, 'w');
if fid < 0
    warning('Unable to write report: %s', reportPath);
    return;
end
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid, '# GA Posterior Structure Comparison (%s)\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Comparison order: static weak structure prior -> posterior-structure-consistency\n\n');
fprintf(fid, '## Run Config\n');
fprintf(fid, '- Trials: %d\n', numberOfTrials);
fprintf(fid, '- baseSeed: %d (fixed=%d)\n\n', baseSeed, useFixedSeed);

fprintf(fid, '## Arm Configs\n');
fprintf(fid, '### Static weak structure prior\n');
fprintf(fid, '- usePosteriorStructureConsistency: %d\n\n', getField(summary.baseOverrides, 'usePosteriorStructureConsistency', false));
fprintf(fid, '### Posterior-structure-consistency\n');
fprintf(fid, '- usePosteriorStructureConsistency: %d\n\n', getField(summary.experimentOverrides, 'usePosteriorStructureConsistency', false));

fprintf(fid, '## Network Disagreement Metrics (mean across trials)\n');
fprintf(fid, '| Arm | OSPA | RMSE | Cardinality |\n');
fprintf(fid, '|:----|-----:|-----:|------------:|\n');
fprintf(fid, '| static weak structure prior | %.6f | %.6f | %.6f |\n', ...
    summary.consensus.ospaBase, summary.consensus.posBase, summary.consensus.cardBase);
fprintf(fid, '| posterior-structure-consistency | %.6f | %.6f | %.6f |\n\n', ...
    summary.consensus.ospaAdaptive, summary.consensus.posAdaptive, summary.consensus.cardAdaptive);

fprintf(fid, '## Aggregated Local Metrics (mean across sensors)\n');
fprintf(fid, '- E-OSPA: %.6f -> %.6f\n', mean(summary.local.eOspaBase), mean(summary.local.eOspaAdaptive));
fprintf(fid, '- RMSE: %.6f -> %.6f\n', mean(summary.local.rmseBase), mean(summary.local.rmseAdaptive));
end

function value = extractScalar(values)
value = values(1);
end

function value = getField(s, fieldName, defaultValue)
if nargin < 3
    defaultValue = [];
end
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function merged = mergeStructFields(base, overrides)
merged = base;
if nargin < 2 || ~isstruct(overrides) || isempty(fieldnames(overrides))
    return;
end
fields = fieldnames(overrides);
for i = 1:numel(fields)
    merged.(fields{i}) = overrides.(fields{i});
end
end

function projectRoot = resolveProjectRoot(scriptDir)
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = scriptDir;
for k = 1:6
    if exist(fullfile(projectRoot, 'setPath.m'), 'file')
        return;
    end
    parent = fileparts(projectRoot);
    if isempty(parent) || strcmp(parent, projectRoot)
        break;
    end
    projectRoot = parent;
end
end
