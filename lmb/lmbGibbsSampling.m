function [r, W, V] = lmbGibbsSampling(associationMatrices, numberOfSamples)
% LMBGIBBSSAMPLING -- Determine posterior existence probabilities and association weights using a Gibbs sampler
%  [r, W, V] = lmbGibbsSampling(associationMatrices, numberOfSamples)
%
%   This function determines each object's posterior existence and marginal
%   association probabilities using Gibbs sampling. This function is a bit
%   more optimised for Matlab.
%   文件导读：
%       LMB 更新的随机关联后端。它从合法 object-to-measurement assignment
%       中采样，去重后再把样本频率和 likelihood 转换成存在概率与边缘
%       关联概率。
%
%   See also runLmbFilter, generateLmbAssociationMatrices,
%       computePosteriorLmbSpatialDistributions, lmbMurtysAlgorithm, 
%       loopyBeliefPropagation, lmbGibbsFrequencySampling
%
%   Inputs
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       numberOfSamples - double. The number of Gibbs samples we want to
%           generate.
%
%   Output
%       r - array. Each object's posterior existence probability.
%       W - array. An array of marginal association probabilities, where
%           each row is an object's marginal association probabilities.
%       V - array. An array of association events, where each row is
%           assignment of objects to measurements.

%% 1. 初始化采样状态和样本缓存
[n, m] = size(associationMatrices.P);
% Association vectors
[v, w] = initialiseGibbsAssociationVectors(associationMatrices.C);
V = zeros(numberOfSamples, n);
%% 2. Gibbs 采样：逐轮生成一个合法关联事件
for i = 1:numberOfSamples
    %% 2.1 生成一个新的 Gibbs sample
    [v, w] = generateGibbsSample(associationMatrices.P, v, w);
    %% 2.2 存储 sample；后面会统一去重
    V(i, :) = v;
end
%% 3. 从样本集合估计边缘关联分布
% 只保留 distinct samples，避免重复 assignment 占用后续计算。
V = unique(V, 'rows');
% 将 assignment event 转成每个 object 的 0..m 关联指示张量。
W = repmat(V, 1, 1, m+1) == reshape(0:m, 1, 1, m+1);
J = reshape(associationMatrices.L(n * V + (1:n)), size(V, 1), n);
L = permute(sum(prod(J, 2) .* W, 1), [2 1 3]);
Sigma = reshape(L, n, m+1);
% 归一化得到 Tau，再拆成 r 和 W。
Tau = (Sigma .* associationMatrices.R) ./ sum(Sigma, 2);
%% 4. 输出 posterior existence probability 和 marginal association probability
r = sum(Tau, 2);
W =  Tau ./ r;
end
