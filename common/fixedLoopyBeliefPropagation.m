function [r, W] = fixedLoopyBeliefPropagation(associationMatrices, maximumNumberOfLbpIterations)
% FIXEDLOOPYBELIEFPROPAGATION -- Determine posterior existence probabilities and association weights using loopy belief propagation
%   [r, W] = fixedLoopyBeliefPropagation(associationMatrices, epsilon)
%
%   This function determines each object's posterior existence and marginal
%   association probabilities using loopy belief propagation (LBP).
%   This algorithm uses a fixed number of iteratrions, and it is only used
%   to verify the LMB filter's asymptotic computational complexity.
%   文件导读：
%       固定迭代次数的 LBP 版本，只用于复杂度或运行时对照。正常滤波
%       推荐使用 loopyBeliefPropagation，因为它会按收敛阈值提前停止。
%
%   See also runLmbFilter, generateLmbAssociationMatrices,
%   computePosteriorLmbSpatialDistributions, lmbGibbsSampling,
%   lmbMurtysAlgorithm, lmbFilterTimeTrials, loopyBeliefPropagation
%
%   Inputs
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       maximumNumberOfLbpIterations - integer. Maximum allowable number
%           of LBP iterations.
%
%   Output
%       r - array. Each object's posterior existence probability.
%       W - array. An array of marginal association probabilities, where
%           each row is an object's marginal association probabilities.

%% 1. 初始化消息
SigmaMT = ones(size(associationMatrices.Psi));
%% 2. 固定轮数 LBP 迭代：不做收敛提前停止
for i = 1:maximumNumberOfLbpIterations
    % object -> measurement 消息。
    B = associationMatrices.Psi .* SigmaMT;
    SigmaTM = associationMatrices.Psi ./ (-B + sum(B, 2) + 1);
    % measurement -> object 消息。
    SigmaMT = 1./ (-SigmaTM + sum(SigmaTM, 1) + 1);
end
Gamma = [associationMatrices.phi B .* associationMatrices.eta];
q = sum(Gamma, 2);
%% 3. 恢复边缘关联概率和存在概率
W = Gamma ./ q;
r = q ./ (associationMatrices.eta + q - associationMatrices.phi);
end
