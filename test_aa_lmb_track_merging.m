function test_aa_lmb_track_merging()
% TEST_AA_LMB_TRACK_MERGING - Regression tests for Bernoulli-AA spatial weights.

clc;
setPath;

fprintf('Test 1: AA spatial mixture is weighted by local existence probability\n');
model = buildModel([0.5, 0.5], [0.5, 0.5], 2);
objects = aaLmbTrackMerging(buildTwoSensorObjectSet(0.95, 0.05), model);
assert(abs(objects(1).r - 0.5) < 1e-12);
assert(abs(objects(1).w(1) - 0.95) < 1e-12);
assert(abs(objects(1).w(2) - 0.05) < 1e-12);

fprintf('Test 2: low-existence component cannot dominate through spatial weight alone\n');
model = buildModel([0.2, 0.8], [0.5, 0.5], 2);
objects = aaLmbTrackMerging(buildTwoSensorObjectSet(0.90, 0.05), model);
expectedFirstWeight = (0.2 * 0.90) / (0.2 * 0.90 + 0.8 * 0.05);
assert(abs(objects(1).w(1) - expectedFirstWeight) < 1e-12);
assert(objects(1).mu{1}(1) == 0);

fprintf('AA-LMB track-merging tests passed.\n');
end

function model = buildModel(spatialWeights, existenceWeights, maxComponents)
model = struct();
model.numberOfSensors = 2;
model.maximumNumberOfGmComponents = maxComponents;
model.aaSensorWeights = ones(1, 2) / 2;
model.aaSpatialWeights = spatialWeights;
model.aaExistenceWeights = existenceWeights;
end

function distributions = buildTwoSensorObjectSet(r1, r2)
distributions = cell(1, 2);
distributions{1} = buildObject(r1, [0; 0; 0; 0]);
distributions{2} = buildObject(r2, [100; 0; 0; 0]);
end

function objects = buildObject(existence, state)
objects = struct( ...
    'r', existence, ...
    'numberOfGmComponents', 1, ...
    'w', 1, ...
    'mu', {{state}}, ...
    'Sigma', {{eye(4)}});
end
