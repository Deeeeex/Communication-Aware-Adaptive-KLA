function test_state_dependent_quality_false_targets_compare()
% TEST_STATE_DEPENDENT_QUALITY_FALSE_TARGETS_COMPARE - Smoke test for the new scenario.

clc;
setPath;
addpath('RUN/GA');

scenarioOverrides = struct();
scenarioOverrides.simulationLength = 12;
scenarioOverrides.clutterRates = 2 * ones(1, 8);
scenarioOverrides.targetBirthInterval = 4;
scenarioOverrides.targetBirthStates = [ ...
     55,  65,  55,  70;
     55,   5, -55, -10;
    -0.5, -0.6, -0.5, -0.6;
    -0.3,  0.0,  0.3,  0.0];
scenarioOverrides.falseTargetConfig = struct( ...
    'birthStates', [20; 45; 0.2; -0.4], ...
    'startTimes', 3, ...
    'endTimes', 12, ...
    'detectionScale', 0.85, ...
    'maxDetectionProbability', 0.75);

[reportPath, summary] = runMultisensorFilters_formation_4plus4_StateDependentQualityFalseTargetsCompare( ...
    1, 9, true, false, struct(), scenarioOverrides);

assert(isempty(reportPath));
assert(isfield(summary, 'scenario'));
assert(isfield(summary, 'falseTargets'));
assert(isfield(summary, 'consensus'));
assert(isfield(summary, 'local'));
assert(isfield(summary, 'delta'));
assert(summary.scenario.numberOfSensors == 8);
assert(summary.scenario.numberOfTrueTargets == 4);
assert(summary.scenario.numberOfFalseTargets == 1);
assert(summary.falseTargets.meanTotal > 0);
assert(isfield(summary.scenario.sensorQuality, 'enabled'));
assert(summary.scenario.sensorQuality.enabled);
assert(isfield(summary.consensus, 'ospaBase'));
assert(isfield(summary.consensus, 'ospaAdaptive'));
assert(isfield(summary.local, 'eOspaBase'));
assert(isfield(summary.local, 'cardErrAdaptive'));
assert(numel(summary.local.eOspaBase) == 8);
assert(numel(summary.local.rmseAdaptive) == 8);

fprintf('State-dependent quality false-target comparison smoke test passed.\n');
end
