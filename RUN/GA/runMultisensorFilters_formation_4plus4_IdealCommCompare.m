function [reportPath, summary] = runMultisensorFilters_formation_4plus4_IdealCommCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, adaptiveFusionOverrides)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_IDEALCOMMCOMPARE
% Run the paper-facing GA arm set under ideal communication.
%
% The arm order matches the manuscript tables:
%   Fixed Metropolis -> PD-weighted GA -> FID-FIA-weighted GA ->
%   Balanced mode -> Cardinality-critical mode.

close all; clc;
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
projectRoot = resolveProjectRoot(scriptDir);
addpath(projectRoot);
addpath(fullfile(projectRoot, 'RUN', 'GA'));
setPath;

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 50;
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

commConfigOverrides = struct();
commConfigOverrides.level = 0;
commConfigOverrides.globalMaxMeasurementsPerStep = inf;
commConfigOverrides.linkModel = 'fixed';
commConfigOverrides.pDrop = 0.0;
commConfigOverrides.pDropBySensor = zeros(1, 8);
commConfigOverrides.maxOutageNodes = 0;

[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, commConfigOverrides, writeReport, ...
    'fidFiaExistenceRefinement', adaptiveFusionOverrides, []);

summary.experiment = 'idealCommunication';
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
