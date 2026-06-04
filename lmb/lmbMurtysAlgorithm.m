function [r, W, V] = lmbMurtysAlgorithm(associationMatrices, numberOfAssignments)
% LMBMURTYSALGORITHM -- Determine posterior existence probabilities and association weights using Murty's algorithm
%   [r, W, V] = lmbMurtysAlgorithm(associationMatrices, numberOfAssignments)
%
%   This function determines each object's posterior existence and marginal
%   association probabilities using Murty's algorithm. This function uses
%   Vo et al.'s code.
%   文件导读：
%       LMB 更新的 K-best assignment 后端。它调用 Murty 算法生成若干最可能
%       的关联事件，再把这些事件转成存在概率和边缘关联概率。通常比 LBP
%       更贵，适合作小规模基准或对照。
%
%   See also runLmbFilter, generateLmbAssociationMatrices,
%   computePosteriorLmbSpatialDistributions, lmbGibbsSampling,
%   loopyBeliefPropagation
%
%   Inputs
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       numberOfAssignments - double. The number of association events we
%           want Murty's algorithm to generate
%
%   Output
%       r - array. Each object's posterior existence probability.
%       W - array. An array of marginal association probabilities, where
%           each row is an object's marginal association probabilities.
%       V - array. An array of association events, where each row is
%           assignment of objects to measurements.

[n, m] = size(associationMatrices.C);
%% 1. 用 Murty 算法生成 K 个最可能 assignment
V = murtysAlgorithmWrapper(associationMatrices.C, numberOfAssignments);
%% 2. 将 K-best assignments 转成边缘关联分布
W = repmat(V, 1, 1, m+1) == reshape(0:m, 1, 1, m+1);
J = reshape(associationMatrices.L(n * V + (1:n)), size(V, 1), n);
L = permute(sum(prod(J, 2) .* W, 1), [2 1 3]);
Sigma = reshape(L, n, m+1);
% 归一化得到 Tau，再拆成 r 和 W。
Tau = (Sigma .* associationMatrices.R) ./ sum(Sigma, 2);
%% 3. 输出 posterior existence probability 和 marginal association probability
r = sum(Tau, 2);
W =  Tau ./ r;
end
