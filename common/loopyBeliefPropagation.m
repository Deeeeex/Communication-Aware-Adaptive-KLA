function [r, W] = loopyBeliefPropagation(associationMatrices, epsilon, maximumNumberOfLbpIterations)
% LOOPYBELIEFPROPAGATION -- Determine posterior existence probabilities and association weights using loopy belief propagation
%   [r, W] = loopyBeliefPropagation(associationMatrices, epsilon, maximumNumberOfLbpIterations)
%
%   This function determines each object's posterior existence and marginal
%   association probabilities using loopy belief propagation (LBP).
%   文件导读：
%       LMB 更新中默认使用的快速关联求解器。输入是关联矩阵构造函数
%       生成的 associationMatrices，输出是每个 Bernoulli 的 posterior
%       existence probability 和每条测量的边缘关联概率。
%
%   See also runLmbFilter, generateLmbAssociationMatrices,
%   computePosteriorLmbSpatialDistributions, lmbGibbsSampling,
%   lmbMurtysAlgorithm
%
%   Inputs
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       epsilon - double. The convergence tolerance for the LBP aglorithm.
%       maximumNumberOfLbpIterations - integer. Maximum allowable number
%           of LBP iterations.
%
%   Output
%       r - (n, 1) array. Each object's posterior existence probability.
%       W - (n, m) array. An array of marginal association probabilities, where
%           each row is an object's marginal association probabilities.

%% 1. 初始化 measurement-to-object 消息
SigmaMT = ones(size(associationMatrices.Psi));
notConverged = true;
counter = 0;
%% 2. LBP 迭代：object cluster 和 measurement cluster 之间交替传消息
while notConverged
    % 缓存上一轮消息，用于收敛判断。
    SigmaMTOld = SigmaMT;
    % object -> measurement 消息。
    B = associationMatrices.Psi .* SigmaMT;
    SigmaTM = associationMatrices.Psi ./ (-B + sum(B, 2) + 1);
    % measurement -> object 消息。
    SigmaMT = 1./ (-SigmaTM + sum(SigmaTM, 1) + 1);
    % 最大消息变化量低于 epsilon 或达到最大迭代次数时停止。
    counter = counter + 1;
    delta = abs(SigmaMT - SigmaMTOld);
    notConverged = (max(delta(:)) > epsilon) && (counter < maximumNumberOfLbpIterations);
end
Gamma = [associationMatrices.phi B .* associationMatrices.eta];
q = sum(Gamma, 2);
%% 3. 从最终消息中恢复边缘关联概率和存在概率
W = Gamma ./ q;
r = q ./ (associationMatrices.eta + q - associationMatrices.phi);
end
