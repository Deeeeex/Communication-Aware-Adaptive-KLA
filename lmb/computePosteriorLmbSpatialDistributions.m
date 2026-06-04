function objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model)
% COMPUTEPOSTERIORLMBSPATIALDISTRIUBUTIONS -- Complete the LMB filter's measurement update
%    objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model)
%
%   This function computes each object's posterior spatial distrubtion. 
%   文件导读：
%       posterior Gaussian mixture 组装步骤。关联求解器只给出存在概率 r
%       和边缘关联概率 W；本函数把这些概率乘到预先计算好的漏检/检测
%       Gaussian component 上，并完成 mixture pruning 和 component 数量上限控制。
%
%   See also generateModel, runLmbFilter, lmbPredictionStep, 
%            loopyBeliefPropagation, generateLmbAssociationMatrices
%
%   Inputs
%       objects - struct. A struct containing the prior LMB's Bernoulli
%           components. This struct is produced by lmbPredictionStep.
%       r - array. Each object's posterior existence probability.
%       W - array. An array of marginal association probabilities, where
%           each row is an object's marginal association probabilities.
%       posteriorParameters - struct. A struct whose fields are an object's
%           posterior spatial distribution parameters.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       objects - struct. A struct containing the posterior LMB's Bernoulli
%           components.

for i = 1:numel(objects)
    %% 1. 写回该 Bernoulli 的 posterior existence probability
    objects(i).r = r(i);
    %% 2. 用边缘关联概率重加权所有漏检/检测 Gaussian components
    numberOfPosteriorComponents = numel(posteriorParameters(i).w);
    posteriorWeights = reshape(W(i, :)' .* posteriorParameters(i).w, 1, numberOfPosteriorComponents);
    posteriorWeights = posteriorWeights ./ sum(posteriorWeights);
    %% 3. mixture reduction：先按权重排序，再删除很小的 component
    [posteriorWeights, sortedIndices] = sort(posteriorWeights, 'descend');
    % Discard insignificant components
    significantComponents = posteriorWeights > model.gmWeightThreshold;
    significantWeights = posteriorWeights(significantComponents);
    objects(i).w = significantWeights ./ sum(significantWeights);
    sortedIndices = sortedIndices(significantComponents);
    objects(i).numberOfGmComponents = numel(objects(i).w);
    % 如果 component 仍太多，就只保留权重最高的前若干个。
    if (objects(i).numberOfGmComponents > model.maximumNumberOfGmComponents)
        objects(i).w = objects(i).w(1:model.maximumNumberOfGmComponents);
        objects(i).w = objects(i).w ./ sum(objects(i).w);
        sortedIndices = sortedIndices(1:model.maximumNumberOfGmComponents);
        objects(i).numberOfGmComponents = model.maximumNumberOfGmComponents;
    end
    %% 4. 写回保留下来的 Gaussian 均值和协方差
    objects(i).mu = reshape(posteriorParameters(i).mu(sortedIndices), 1, objects(i).numberOfGmComponents);
    objects(i).Sigma = reshape(posteriorParameters(i).Sigma(sortedIndices), 1, objects(i).numberOfGmComponents);
end

end
