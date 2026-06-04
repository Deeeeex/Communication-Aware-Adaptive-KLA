function test_ideal_comm_compare()
% TEST_IDEAL_COMM_COMPARE - Smoke test for ideal-communication GA comparison

clc;
setPath;
addpath('RUN/GA');

[reportPath, summary] = runMultisensorFilters_formation_4plus4_IdealCommCompare(1, 1, true, false);

assert(isempty(reportPath));
assert(isfield(summary, 'consensus'));
assert(isfield(summary, 'local'));
assert(isfield(summary, 'delta'));
assert(isfield(summary.consensus, 'ospaBase'));
assert(isfield(summary.consensus, 'ospaAdaptive'));
assert(isfield(summary.consensus, 'posBase'));
assert(isfield(summary.consensus, 'cardBase'));
assert(isfield(summary.local, 'eOspaBase'));
assert(isfield(summary.local, 'hOspaBase'));
assert(isfield(summary.local, 'rmseBase'));
assert(numel(summary.local.eOspaBase) == 8);
assert(numel(summary.local.hOspaAdaptive) == 8);
assert(all(abs(summary.meanPDropBySensor) < 1e-12));

fprintf('Ideal communication comparison smoke test passed.\n');
end
