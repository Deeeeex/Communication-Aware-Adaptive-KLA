function posteriorHypotheses = determinePosteriorHypothesisParameters(V, L, posteriorParameters, priorHypothesis)
% DETERMINEPOSTERIORHYPOTHESISPARAMETERS -- Determine a parameters for a new set of posterior LMBM hypotheses.
%   posteriorHypotheses = determinePosteriorHypothesisParameters(V, L, posteriorParameters, priorHypothesis)
%
%   Determine a parameters for a new set of posterior LMBM hypotheses.
%   These hypotheses will have unnormalised hypothesis weights.
%   文件导读：
%       将 Gibbs 或 Murty 生成的关联事件转换为 LMBM posterior hypotheses。
%       这里会把 association likelihood 写入 hypothesis weight，并根据
%       关联事件选择存在概率、均值和协方差。
%
%   See also runLmbmFilter, generateLmbmAssociationMatrices, lmbmGibbsSampling
%
%   Inputs
%       V - array. An array of distinct association events, where each row of the
%           array is an association event. See lmbmGibbsSampling.
%       L - array. An array of marginal log likelihood ratios. See
%           generateLmbmGibbsMatrices.
%       posteriorParameters - struct. A struct whose fields are a
%           hypothesis posterior distribution parameters. See
%           generateLmbmGibbsMatrices.
%       priorHypothesis - struct. A struct containing the prior LMBM hypotheses' Bernoulli components.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       posteriorHypotheses - struct. A struct containing posterior LMBM
%           hypotheses, but with unnormalised hypothesis weights.

%% 1. 准备输出 hypotheses；每个关联事件对应一个 posterior hypothesis
numberOfObjects = numel(priorHypothesis.r);
eta = 1:numberOfObjects;
numberOfPosteriorHypotheses = size(V, 1);
priorHypothesis.r = posteriorParameters.r;
posteriorHypotheses = repmat(priorHypothesis, 1, numberOfPosteriorHypotheses);
%% 2. 遍历关联事件，写入 hypothesis weight 和 component 参数
for i = 1:numberOfPosteriorHypotheses
    % 当前关联事件。
    v = V(i, :);
    % 将 object/measurement 关联事件转成 posteriorParameters 的线性索引。
    ell = numberOfObjects * v + eta;
    % v>0 表示该 object 生成了某条测量，否则是 missed detection。
    generatedMeasurement = v > 0;
    % hypothesis weight = prior weight + 当前关联事件 log likelihood。
    posteriorHypotheses(i).w = log(priorHypothesis.w) + sum(L(ell));
    % Existence probabilities
    posteriorHypotheses(i).r(generatedMeasurement, :) = 1;
    % Means
    posteriorHypotheses(i).mu = posteriorParameters.mu(ell);
    % Covariance
    posteriorHypotheses(i).Sigma(generatedMeasurement) = posteriorParameters.Sigma(generatedMeasurement);
end
end
