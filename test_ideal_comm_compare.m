function test_ideal_comm_compare()
% TEST_IDEAL_COMM_COMPARE - Lightweight smoke test for ideal-communication GA setup

clc;
setPath;
addpath('RUN/GA');

[reportPath, summary] = runMultisensorFilters_formation_4plus4_IdealCommCompare(0, 1, true, false);

assert(isempty(reportPath));
expectedArmNames = {'Fixed Metropolis', 'PD-weighted GA', 'FID-FIA-weighted GA', ...
    'Balanced mode', 'Cardinality-critical mode'};
assert(isequal(summary.armNames, expectedArmNames));
assert(strcmp(summary.experiment, 'idealCommunication'));
assert(isfield(summary, 'consensus'));
assert(isfield(summary, 'local'));
assert(isfield(summary.consensus, 'ospa'));
assert(isfield(summary.consensus, 'pos'));
assert(isfield(summary.consensus, 'card'));
assert(isfield(summary.local, 'eOspa'));
assert(isfield(summary.local, 'hOspa'));
assert(isfield(summary.local, 'rmse'));
assert(numel(summary.consensus.ospa) == 5);
assert(size(summary.local.eOspa, 1) == 8);
assert(size(summary.local.eOspa, 2) == 5);
assert(summary.commConfig.level == 0);
assert(isinf(summary.commConfig.globalMaxMeasurementsPerStep));
assert(abs(summary.commConfig.pDrop) < 1e-12);
assert(all(abs(summary.commConfig.pDropBySensor) < 1e-12));

fprintf('Ideal communication comparison smoke test passed.\n');
end
