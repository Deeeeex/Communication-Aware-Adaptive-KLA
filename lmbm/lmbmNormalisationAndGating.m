function [hypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model)
% LMBMNORMALISATIONANDGATING -- Gate the number of posterior hypotheses, and their parameters.
%    [hypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model)
%
%   This function discards unlikely posterior parameters, and discards
%   Bernoulli with low existence probabilities from each hypothesis.
%   文件导读：
%       LMBM hypothesis 管理步骤。它负责归一化 global hypothesis 权重、
%       删除低权重 hypotheses、限制 hypothesis 数量，并根据总存在概率
%       返回 component/trajectory 的保留 mask。
%
%   See also runLmbmFilter
%
%   Inputs
%       posteriorHypotheses - struct. A struct containing posterior LMBM
%           hypotheses, but with unnormalised hypothesis weights.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       hypotheses - struct. A struct containing likely posterior LMBM
%           hypotheses, but with normalised hypothesis weights.
%       objectsLikelyToExist - array. An array of boolean indicating which objects
%           have been kept, and which have been discarded.

%% 1. 归一化 posterior hypothesis 权重
logW = [posteriorHypotheses.w];
maxW = max(logW, [], 2);
w = exp(logW - maxW) ./ sum(exp(logW - maxW));
%% 2. 删除低权重 hypotheses
likelyHypotheses = w > model.posteriorHypothesisWeightThreshold;
hypotheses = posteriorHypotheses(likelyHypotheses);
w = w(likelyHypotheses) ./ sum(w(likelyHypotheses));
%% 3. 按 hypothesis 权重降序排序
[w, sortedIndices] = sort(w, 'descend');
hypotheses = hypotheses(sortedIndices);
%% 4. 如果 hypothesis 数量过多，只保留最高权重的一批
numberOfHypotheses = numel(w);
if (numberOfHypotheses > model.maximumNumberOfPosteriorHypotheses)
    w = w(1:model.maximumNumberOfPosteriorHypotheses);
    w = w ./ sum(w);
    hypotheses = hypotheses(1:model.maximumNumberOfPosteriorHypotheses);
    numberOfHypotheses = model.maximumNumberOfPosteriorHypotheses;
end
%% 5. 汇总每个 component 的总存在概率，并同步剪枝所有 hypotheses
r = sum(w .* [hypotheses.r], 2);
objectsLikelyToExist = r > model.existenceThreshold;
for i = 1:numberOfHypotheses
    % Truncate components
    hypotheses(i).birthLocation = hypotheses(i).birthLocation(objectsLikelyToExist);
    hypotheses(i).birthTime = hypotheses(i).birthTime(objectsLikelyToExist);
    hypotheses(i).w = w(i);
    hypotheses(i).r = hypotheses(i).r(objectsLikelyToExist, :);
    hypotheses(i).mu = hypotheses(i).mu(objectsLikelyToExist);
    hypotheses(i).Sigma = hypotheses(i).Sigma(objectsLikelyToExist);
end
%% 6. 极端情况下兜底，避免返回空 hypothesis 集合
if (numberOfHypotheses == 0)
    hypotheses = model.hypotheses;
end
end
