function [associationMatrices, posteriorParameters] = generateLmbmAssociationMatrices(hypothesis, z, model)
% GENERATELMBMASSOCIATIONMATRICES -- Compute the association matrices required by the data association algorithms.
%   [gibbsParameters, posteriorParameters] = generateLmbGibbsMatrices(objects, z, model)
%
%   This function computes the association matrices required by the LBP.
%   Gibbs sampler, and Murty's algorithms. It also determines the measurement-updated components that
%   are used to determine each object's posterior spatial distribution.
%   文件导读：
%       LMBM 的关联矩阵前端。它和 LMB 关联矩阵构造逻辑相似，但输入是
%       单个 global hypothesis，并输出用于 hypothesis branching 的
%       likelihood 表和 posterior 参数。
%
%   See also runLmbmFilter, generateModel, lmbmGibbsSampling,
%
%   Inputs
%       hypothesis - struct. A struct containing the prior LMBM hypothesis' Bernoulli components.
%       z - cell array. A cell array of measurements for the
%           current time-step.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       posteriorParameters - struct. A struct whose fields are a
%           hypothesis posterior distribution parameters.

%% 1. 分配 likelihood 矩阵和 posterior 参数容器
numberOfObjects = numel(hypothesis.r);
numberOfMeasurements = numel(z);
% Log likelihood matrix
R = zeros(numberOfObjects, numberOfMeasurements);
% Auxiliary variables
phi = (1 - model.detectionProbability) * hypothesis.r;
eta = 1 - model.detectionProbability * hypothesis.r;
% Updated components for the objects' posterior spatial distributions
posteriorParameters.r = phi ./ eta;
posteriorParameters.mu = cell(numberOfObjects, numberOfMeasurements + 1);
posteriorParameters.Sigma = hypothesis.Sigma;
%% 2. 对每个 Bernoulli component 计算漏检和每条测量对应的 posterior 参数
for i = 1:numberOfObjects
    % Missed detection event
    posteriorParameters.mu{i, 1} = hypothesis.mu{i};
    % 为当前 component 预先计算创新协方差和 Kalman gain。
    muZ = model.C * hypothesis.mu{i};
    Z = model.C * hypothesis.Sigma{i} * model.C' + model.Q;
    logGaussianNormalisingConstant = - (0.5 * model.zDimension) * log(2 * pi) - 0.5 * log(det(Z));
    logLikelihoodRatioTerms = log(hypothesis.r(i)) + log(model.detectionProbability) - log(model.clutterPerUnitVolume);
    ZInv = inv(Z);
    K = hypothesis.Sigma{i} * model.C' * ZInv;
    posteriorParameters.Sigma{i} = (eye(model.xDimension) - K * model.C) * hypothesis.Sigma{i};
    % 遍历每条测量，计算 likelihood ratio 并保存 posterior mean。
    for j = 1:numberOfMeasurements
        % 当前 component 生成 measurement j 的 log likelihood ratio。
        nu = z{j} - muZ;
        gaussianLogLikelihood = logGaussianNormalisingConstant - 0.5 * nu' * ZInv * nu;
        R(i, j) = logLikelihoodRatioTerms + gaussianLogLikelihood;
        % 当前测量条件下的 posterior mean。
        posteriorParameters.mu{i, j+1} = hypothesis.mu{i} + K * nu;
    end
end
%% 3. 输出 Gibbs/Murty 使用的关联矩阵字段
RLinear = exp(R);
% Gibbs sampler association matrices
associationMatrices.P = RLinear ./ (RLinear + eta);
associationMatrices.L = [log(eta) R];
% Murty's algorithm cost matrix
associationMatrices.C = -R;
end
