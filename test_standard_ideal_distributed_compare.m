function test_standard_ideal_distributed_compare()
% TEST_STANDARD_IDEAL_DISTRIBUTED_COMPARE - Smoke test for distributed ideal GA comparison

clc;
setPath;
addpath('RUN/IDEAL');

[reportPath, summary] = runStandardFixedIdealDistributedCompare(1, 1, true, false);

assert(isempty(reportPath));
assert(isfield(summary, 'config'));
assert(isfield(summary, 'baseline'));
assert(isfield(summary, 'adaptive'));
assert(isfield(summary, 'consensus'));
assert(isfield(summary, 'structure'));
assert(summary.config.numberOfSensors == 4);
assert(strcmp(summary.config.topologyName, 'ring'));
assert(all(abs(summary.meanPDropBySensor) < 1e-12));
assert(numel(summary.structure.spatialPriors) == 4);
assert(any(abs(summary.structure.spatialPriors{1} - mean(summary.structure.spatialPriors{1})) > 1e-9));
assert(isfield(summary.consensus, 'ospaBaseline'));
assert(isfield(summary.consensus, 'ospaAdaptive'));

fprintf('Standard ideal distributed comparison smoke test passed.\n');
end
