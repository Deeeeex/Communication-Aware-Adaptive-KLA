function test_standard_ideal_fixed_compare()
% TEST_STANDARD_IDEAL_FIXED_COMPARE - Smoke test for the standard ideal GA/AA scenario

clc;
setPath;
addpath('RUN/IDEAL');

[reportPath, summary] = runMultisensorFilters_standard_fixed_ideal_compare(1, 1, true, false);

assert(isempty(reportPath));
assert(isfield(summary, 'config'));
assert(isfield(summary, 'ga'));
assert(isfield(summary, 'aa'));
assert(isfield(summary.ga, 'eOspa'));
assert(isfield(summary.ga, 'hOspa'));
assert(isfield(summary.ga, 'rmse'));
assert(isfield(summary.aa, 'eOspa'));
assert(isfield(summary.aa, 'hOspa'));
assert(isfield(summary.aa, 'rmse'));
assert(summary.config.numberOfSensors == 4);
assert(strcmp(summary.config.scenarioType, 'Fixed'));
assert(strcmp(summary.config.fusionWeighting, 'Uniform'));
assert(all(abs(summary.meanPDropBySensor) < 1e-12));

fprintf('Standard ideal fixed GA/AA comparison smoke test passed.\n');
end
