function [reportPath, summary] = runMultisensorFilters_formation_4plus4_FiWeightedGaCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, commConfigOverrides, adaptiveFusionOverrides)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_FIWEIGHTEDGACOMPARE
% Evaluate PD-weighted GA and FI-weighted GA baselines in the default
% tiered-link 4+4 formation scenario.

if nargin < 1 || isempty(numberOfTrials)
    numberOfTrials = 3;
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
if nargin < 5 || isempty(commConfigOverrides)
    commConfigOverrides = struct();
end
if nargin < 6 || isempty(adaptiveFusionOverrides)
    adaptiveFusionOverrides = struct();
end

[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, commConfigOverrides, writeReport, ...
    'fiWeightedGa', adaptiveFusionOverrides, []);
end
