function posteriorHypotheses = determineMultisensorPosteriorHypothesisParameters(A, L, posteriorParameters, priorHypothesis)
% DETERMINEMULTISENSORPOSTERIORHYPOTHESISPARAMETERS -- Determine a parameters for a new set of posterior LMBM hypotheses.
%  posteriorHypotheses = determineMultisensorPosteriorHypothesisParameters(V, L, posteriorParameters, priorHypothesis)
%
%   Determine a parameters for a new set of posterior LMBM hypotheses.
%   These hypotheses will have unnormalised hypothesis weights.
%   文件导读：
%       将多传感器关联事件重建为 posterior hypotheses，是单传感器
%       determinePosteriorHypothesisParameters 的多传感器版本。
%
%   See also runLmbmFilter, generateMultisensorLmbmAssociationMatrices, lmbmGibbsSampling
%
%   Inputs
%       A - array. An array of distinct association events, where each row of the
%           array is an association event. See lmbmGibbsSampling.
%       L - array. An array of marginal log likelihood ratios. See
%           generateLmbmGibbsMatrices.
%       posteriorParameters - struct. A struct whose fields are a
%           hypothesis posterior distribution parameters. See
%           generateLmbmGibbsMatrices.
%       priorHypothesis - struct. A struct containing the prior LMBM hypothesis' Bernoulli components.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       posteriorHypotheses - struct. A struct containing posterior LMBM
%           hypotheses, but with unnormalised hypothesis weights.

%% 1. 读取关联张量维度
d = size(L);
m = d(1:end-1) - 1;
numberOfObjects = d(end);
numberOfSensors = length(m);
%% 2. 将采样得到的关联事件整理成线性索引所需格式
eta = reshape(1:numberOfObjects, numberOfObjects, 1);
U = [zeros(numberOfObjects, numberOfSensors) eta];
A = A + 1;
%% 3. 预分配输出 hypotheses
numberOfPosteriorHypotheses = size(A, 1);
posteriorHypotheses = repmat(priorHypothesis, 1, numberOfPosteriorHypotheses);
%% 4. 遍历关联事件，写入 posterior hypothesis 参数
for i = 1:numberOfPosteriorHypotheses
    % 当前 multi-sensor association event。
    U(:, 1:numberOfSensors) = reshape(A(i, :), numberOfObjects, numberOfSensors);
    % 线性索引用于从 L/posteriorParameters 张量取值。
    ell = determineLinearIndex(U, d);
    % hypothesis weight = prior weight + 当前关联事件 log likelihood。
    posteriorHypotheses(i).w = log(priorHypothesis.w) + sum(L(ell));
    % Existence probabilities
    posteriorHypotheses(i).r = posteriorParameters.r(ell);
    % Means
    posteriorHypotheses(i).mu = posteriorParameters.mu(ell);
    % Covariance
    posteriorHypotheses(i).Sigma = posteriorParameters.Sigma(ell);
end
end
%% 根据多传感器关联向量计算张量线性索引
function ell = determineLinearIndex(U, d)
ell = U(:, 1);
Pi = 1;
for i = 2:size(U, 2)
    Pi = Pi * d(i-1);
    ell  = ell  + Pi * (U(:, i) - 1);
end
end
