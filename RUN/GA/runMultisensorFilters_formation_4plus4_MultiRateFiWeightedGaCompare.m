function [reportPath, summary] = runMultisensorFilters_formation_4plus4_MultiRateFiWeightedGaCompare( ...
    numberOfTrials, baseSeed, useFixedSeed, writeReport, commConfigOverrides, adaptiveFusionOverrides)
% RUNMULTISENSORFILTERS_FORMATION_4PLUS4_MULTIRATEFIWEIGHTEDGACOMPARE
% Evaluate the paper-style FI-proportional GA dynamic-weight baseline under
% heterogeneous per-sensor sampling periods.

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

commConfig = struct();
commConfig.enableMultiRate = true;
commConfig.samplingPeriods = [1, 2, 3, 4, 1, 2, 3, 4];
commConfig.samplingPhaseOffsets = zeros(1, 8);
commConfig = mergeStructFields(commConfig, commConfigOverrides);

[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, commConfig, writeReport, ...
    'fiWeightedGa', adaptiveFusionOverrides, []);
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
