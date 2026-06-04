function objects = gaLmbTrackMerging(measurementUpdatedDistributions, model)
% GALMBTRACKMERGING -- Merge the objects' measurement-updated distributions together using the GA-fusion rule.
%   objects = gaLmbTrackMerging(measurementUpdatedDistributions, model)
%
%   Merge the objects' measurement-updated distributions together using the GA-fusion rule.
%   This makes use of a very crude merging algorithm that is actually
%   reasonably accurate. It might be possible to extend this to Gaussian
%   mixtures, or, using the well-space mixture assumptions, apply
%   expectation propagation.
%   文件导读：
%       GA/KLA 融合的权重消费端。动态权重不是在这里计算，而是在这里
%       真正进入融合公式：spatial weights 用于 Gaussian canonical 参数
%       加权，existence weights 用于 Bernoulli existence probability 加权。
%       因此这个文件是理解 branch-decoupled KLA 落地位置的关键：
%       computeAdaptiveFusionWeights 负责产生 gaSpatialWeights 和
%       gaExistenceWeights；本函数负责把它们分别用在空间 density 和
%       existence probability 的几何平均中。
%
%   调用位置：
%       runParallelUpdateLmbFilter 每个时刻先得到 measurementUpdatedDistributions，
%       更新 model.gaSpatialWeights / model.gaExistenceWeights 后，再调用本函数。
%
%   融合顺序：
%       1. 对每个 Bernoulli component / target 单独处理；
%       2. 对每个 sensor 的 Gaussian mixture 做 m-projection，近似成单 Gaussian；
%       3. 用 spatialWeights 在 canonical form 中做几何平均；
%       4. 用 existenceWeights 融合 Bernoulli existence probability；
%       5. 写回单 Gaussian posterior component。
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
for i = 1:numel(objects)
    %% 1. 读取当前目标的空间分支权重和存在分支权重
    % 如果 direct baseline 返回了 target-wise weights，则优先使用目标级权重；
    % 否则使用全局 gaSpatialWeights / gaExistenceWeights。
    % 这一步让 PD/FI direct baseline 能对不同 target 给不同 sensor 权重；
    % 对 Balanced / Cardinality-critical，则通常是每个 target 共享同一组
    % spatial/existence branch weights。
    spatialWeights = resolveObjectWeightVector( ...
        model, 'gaTargetWiseWeights', 'gaSpatialWeights', model.gaSensorWeights, i);
    existenceWeights = resolveObjectWeightVector( ...
        model, 'gaTargetWiseWeights', 'gaExistenceWeights', model.gaSensorWeights, i);
    %% 2. 空间融合：先 m-projection，再在 canonical form 中做几何平均
    % GA/KLA 对 Gaussian density 做加权几何平均。为了避免直接处理 mixture
    % product，这里先把每个传感器的 mixture posterior moment-match 成单 Gaussian，
    % 再把 Gaussian 写成 canonical form：
    %   K = Sigma^{-1}
    %   h = K * mu
    % 多传感器融合就是对 K、h 和 log-normalizer 按 spatialWeights 求和。
    K = zeros(model.xDimension, model.xDimension);
    h = zeros(model.xDimension, 1);
    g = 0;
    for s = 1:model.numberOfSensors
        [nu, T] = mprojection(model.xDimension, measurementUpdatedDistributions{s}(i));
        % 将每个传感器的 posterior 近似成单 Gaussian，再乘上 spatial weight。
        % spatialWeights 越大，该传感器的 precision matrix 对融合结果影响越大。
        KMatched = spatialWeights(s) * inv(T);
        hMatched = KMatched * nu;
        gMatched = -0.5 * nu' * KMatched * nu - 0.5 * spatialWeights(s) * log(det(2*pi*T));
        % Throw it on the pile
        K = K + KMatched;
        h = h + hMatched;
        g = g + gMatched;
    end
    %% 3. 将 canonical form 转回均值/协方差，并计算归一化项 eta
    % eta 是 Gaussian density 几何平均后的积分归一化项。它会进入 existence
    % 融合公式，因为 Bernoulli RFS 的存在概率和空间 density 归一化耦合。
    SigmaGa = inv(K);
    muGa = SigmaGa * h;
    eta = exp(g + 0.5 * muGa' * K * muGa + 0.5 * log(det(2*pi*SigmaGa)));
    %% 4. 存在概率融合：用 existence weights 合并各传感器 Bernoulli r
    % 这里是 branch-decoupled 的第二个落点。existenceWeights 可以不同于
    % spatialWeights，因此 Cardinality-critical 可以在不破坏空间定位权重的
    % 前提下，更强地压低 cardinality/existence 不可靠的传感器。
    numerator = eta;
    partialDenominator = 1;
    for s = 1:model.numberOfSensors
        rS = measurementUpdatedDistributions{s}(i).r;
        numerator = numerator * (rS^(existenceWeights(s)));
        partialDenominator = partialDenominator *  ((1-rS)^(existenceWeights(s)));
    end
    %% 5. 写回融合后的 Bernoulli component
    % GA 融合后每个 component 被压成单 Gaussian。后续 runParallelUpdateLmbFilter
    % 会按 existenceThreshold 剪枝，再做 MAP cardinality/state extraction。
    objects(i).r = numerator / (numerator + partialDenominator);
    objects(i).numberOfGmComponents = 1;
    objects(i).w = 1;
    objects(i).mu = {muGa};
    objects(i).Sigma = {SigmaGa};
end
end

function weights = resolveObjectWeightVector(model, targetWiseFieldName, fieldName, fallback, objectIdx)
% RESOLVEOBJECTWEIGHTVECTOR - 为当前 target 选择一组 sensor weights。
% 优先级：
%   1. target-wise weights：PD/FI direct baseline 可按 target 调整传感器权重；
%   2. branch weights：Balanced/Cardinality-critical 的 spatial/existence 权重；
%   3. fallback：固定 GA sensor weights。
weights = [];
if isfield(model, targetWiseFieldName)
    targetWiseWeights = model.(targetWiseFieldName);
    if size(targetWiseWeights, 1) >= objectIdx && size(targetWiseWeights, 2) == numel(fallback)
        weights = targetWiseWeights(objectIdx, :);
    end
end
if isempty(weights)
    weights = resolveWeightVector(model, fieldName, fallback);
end
weights = normalizeWeightVector(weights, fallback);
end

function weights = resolveWeightVector(model, fieldName, fallback)
% RESOLVEWEIGHTVECTOR - 读取指定 branch 的权重字段，不存在时回退到 fixed weights。
if isfield(model, fieldName)
    weights = model.(fieldName);
else
    weights = fallback;
end
end

function weights = normalizeWeightVector(weights, fallback)
% NORMALIZEWEIGHTVECTOR - 把权重整理成合法概率向量。
% 动态权重可能因为数值问题、维度错位或全零 score 变得不可用；这里统一
% 做非负化、fallback 和归一化，保证 GA 融合拿到的总是 convex weights。
weights = reshape(weights, 1, []);
weights(~isfinite(weights)) = 0;
weights = max(weights, 0);
if numel(weights) ~= numel(fallback) || sum(weights) <= 0
    weights = reshape(fallback, 1, []);
    weights(~isfinite(weights)) = 0;
    weights = max(weights, 0);
end
if sum(weights) <= 0
    weights = ones(1, numel(fallback)) / numel(fallback);
else
    weights = weights / sum(weights);
end
end

%% M-projection：把 Gaussian mixture moment-match 成单 Gaussian，供 GA 融合使用
function [nu, T] = mprojection(n, measurementUpdatedDistribution)
% 计算 m-projection 后的均值。
nu = zeros(n, 1);
for j = 1:measurementUpdatedDistribution.numberOfGmComponents
    nu = nu + measurementUpdatedDistribution.w(j) * measurementUpdatedDistribution.mu{j};
end
% 计算 m-projection 后的协方差。
T = zeros(n, n);
for j = 1:measurementUpdatedDistribution.numberOfGmComponents
    w = measurementUpdatedDistribution.w(j);
    mu = measurementUpdatedDistribution.mu{j} - nu;
    Sigma = measurementUpdatedDistribution.Sigma{j};
    T = T + w * (Sigma + mu * mu');
end
end
