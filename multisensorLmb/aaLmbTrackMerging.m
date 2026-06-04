function objects = aaLmbTrackMerging(measurementUpdatedDistributions, model)
%  AALMBTRACKMERGING -- Merge the objects' measurement-updated distributions together using the AA-fusion rule.
%   objects = aaLmbTrackMerging(measurementUpdatedDistributions, model)
%
%   Merge the objects' measurement-updated distributions together using the AA-fusion rule.
%   文件导读：
%       AA 融合的权重消费端。existence weights 用于线性平均 Bernoulli r；
%       spatial mixture 则按 spatial weight 和 local existence probability
%       共同加权后拼接 Gaussian component。
%       当前主线偏 GA/KLA，但 AA 路径保留了同一套动态权重接口。
%
%   See also generateMultisensorModel, loopyBeliefPropagation, lmbGibbsSampling, lmbMurtysAlgorithm
%
%   Inputs
%       measurementUpdatedDistributions - (1, numberOfSensors) cell array. 
%           Each is an object struct produced by computePosteriorLmbSpatialDistributions.   
%       model - struct. A struct with the fields declared in generateMultisensorModel.
%
%   Output
%       objects - struct. A struct containing the posterior LMB's Bernoulli
%           components.

objects = measurementUpdatedDistributions{1};
%% 1. 读取 AA 的空间/存在分支权重；没有 branch 权重时回退到 aaSensorWeights
spatialWeights = normalizeWeightVector( ...
    resolveWeightVector(model, 'aaSpatialWeights', model.aaSensorWeights), model.aaSensorWeights);
existenceWeights = normalizeWeightVector( ...
    resolveWeightVector(model, 'aaExistenceWeights', model.aaSensorWeights), model.aaSensorWeights);
for i = 1:numel(objects)
    %% 2. Bernoulli-AA：existence 线性平均，spatial mixture 用 r_s 加权
    % For a Bernoulli density, arithmetic averaging gives
    %   r = sum_s alpha_s r_s
    %   p(x) proportional to sum_s alpha_s r_s p_s(x).
    % The previous implementation used only alpha_s for p_s(x), which let
    % low-existence but sharp local posteriors dominate the output Gaussian.
    fusedExistence = 0;
    fusedWeights = [];
    fallbackWeights = [];
    fusedMeans = {};
    fusedCovariances = {};
    for s = 1:model.numberOfSensors
        localObject = measurementUpdatedDistributions{s}(i);
        localExistence = clampProbability(localObject.r);
        localWeights = reshape(localObject.w, 1, []);

        fusedExistence = fusedExistence + existenceWeights(s) * localExistence;
        fusedWeights = horzcat(fusedWeights, spatialWeights(s) * localExistence * localWeights);
        fallbackWeights = horzcat(fallbackWeights, spatialWeights(s) * localWeights);
        fusedMeans = horzcat(fusedMeans, localObject.mu);
        fusedCovariances = horzcat(fusedCovariances, localObject.Sigma);
    end
    objects(i).r = clampProbability(fusedExistence);
    if sum(fusedWeights) <= eps
        fusedWeights = fallbackWeights;
    end
    %% 4. 按 mixture 权重排序并截断，避免 component 数量无限增长
    [~, sortedIndices] = sort(fusedWeights, 'descend');
    numberOfGmComponents = numel(fusedWeights);
    sortedIndices = sortedIndices(1:min(model.maximumNumberOfGmComponents, numberOfGmComponents));
    objects(i).numberOfGmComponents = numel(sortedIndices);
    retainedWeights = fusedWeights(sortedIndices);
    weightSum = sum(retainedWeights);
    if weightSum <= eps
        retainedWeights = ones(1, numel(sortedIndices)) / max(numel(sortedIndices), 1);
    else
        retainedWeights = retainedWeights ./ weightSum;
    end
    objects(i).w = retainedWeights;
    objects(i).mu = fusedMeans(sortedIndices);
    objects(i).Sigma = fusedCovariances(sortedIndices);
end


end

function weights = resolveWeightVector(model, fieldName, fallback)
if isfield(model, fieldName)
    weights = model.(fieldName);
else
    weights = fallback;
end
end

function weights = normalizeWeightVector(weights, fallback)
weights = reshape(weights, 1, []);
fallback = reshape(fallback, 1, []);
weights(~isfinite(weights)) = 0;
weights = max(weights, 0);
if numel(weights) ~= numel(fallback) || sum(weights) <= 0
    weights = fallback;
    weights(~isfinite(weights)) = 0;
    weights = max(weights, 0);
end
if sum(weights) <= 0
    weights = ones(1, numel(fallback)) / numel(fallback);
else
    weights = weights / sum(weights);
end
end

function value = clampProbability(value)
if ~isfinite(value)
    value = 0;
end
value = min(max(value, 0), 1);
end
