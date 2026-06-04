function [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, z, model)
% GENERATELMBASSOCIATIONMATRICES -- Compute the association matrices required for the data association algorithms
%   [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, z, model)
%
%   This function computes the association matrices required by the LBP,
%   Gibbs sampler, and Murty's algorithms. It also determines the measurement-updated components that
%   are used to determine each object's posterior spatial distribution.
%   文件导读：
%       单传感器 measurement update 的前端。它把每个 Bernoulli component
%       对每条测量的 likelihood ratio、漏检项和 Kalman 更新后的 Gaussian
%       component 统一算好，再输出给 LBP/Gibbs/Murty 共用的矩阵字段。
%
%   See also runLmbFilter, generateModel, loopyBeliefPropagation, lmbGibbsSampling, lmbMurtysAlgorithm
%
%   Inputs
%       objects - struct. A struct containing the prior LMB's Bernoulli components.
%       z - cell array. A cell array of measurements for the
%           current time-step.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       posteriorParameters - struct. A struct whose fields are an object's
%           posterior spatial distribution parameters.
%

%% 1. 分配关联矩阵和 posterior component 容器
numberOfObjects = numel(objects);
numberOfMeasurements = numel(z);
% Auxillary matrices
L = zeros(numberOfObjects, numberOfMeasurements);
phi = zeros(numberOfObjects, 1);
eta = zeros(numberOfObjects, 1);
% Updated components for the objects' posterior spatial distributions
posteriorParameters.w = [];
posteriorParameters.mu = {};
posteriorParameters.Sigma = {};
posteriorParameters = repmat(posteriorParameters, 1, numberOfObjects);
%% 2. 对每个 Bernoulli component 计算漏检项和每条量测的更新项
for i = 1:numberOfObjects
    % 第 1 行是 missed-detection component，后面每一行对应一条测量。
    posteriorParameters(i).w = repmat(log(objects(i).w * (1 - model.detectionProbability)), numberOfMeasurements + 1, 1);
    posteriorParameters(i).mu  = repmat(objects(i).mu, numberOfMeasurements + 1, 1);
    posteriorParameters(i).Sigma = repmat(objects(i).Sigma, numberOfMeasurements + 1, 1);
    % Populate auxiliary LBP parameters
    phi(i) = (1 -  model.detectionProbability) * objects(i).r;
    eta(i) = 1 - model.detectionProbability * objects(i).r;
    %% 2.1 遍历该 Bernoulli 的 Gaussian mixture component
    for j = 1:objects(i).numberOfGmComponents
        % 对当前 Gaussian component 预先计算创新协方差和 Kalman gain。
        muZ = model.C * objects(i).mu{j};
        Z = model.C * objects(i).Sigma{j} * model.C' + model.Q;
        logGaussianNormalisingConstant = - (0.5 * model.zDimension) * log(2 * pi) - 0.5 * log(det(Z));
        logLikelihoodRatioTerms = log(objects(i).r) + log(model.detectionProbability) + log(objects(i).w(j)) - log(model.clutterPerUnitVolume);
        ZInv = inv(Z);
        K = objects(i).Sigma{j} * model.C' * ZInv;
        SigmaUpdated = (eye(model.xDimension) - K * model.C) * objects(i).Sigma{j};
        % 将当前 Gaussian component 对所有测量的贡献累加到 L 和 posteriorParameters。
        for k = 1:numberOfMeasurements
            % L(i,k) 是 object i 生成 measurement k 的边缘 likelihood ratio。
            nu = z{k} - muZ;
            gaussianLogLikelihood = logGaussianNormalisingConstant - 0.5 * nu' * ZInv * nu;
            L(i, k) = L(i, k) + exp(logLikelihoodRatioTerms + gaussianLogLikelihood);
            % 保存 measurement k 条件下的 posterior Gaussian component。
            posteriorParameters(i).w(k+1, j) = log(objects(i).w(j)) + gaussianLogLikelihood + log(model.detectionProbability) - log(model.clutterPerUnitVolume);
            posteriorParameters(i).mu{k+1, j} = objects(i).mu{j} + K * nu;
            posteriorParameters(i).Sigma{k+1, j} = SigmaUpdated;
        end
    end
    % 每个“漏检/检测事件”内部都要归一化 Gaussian mixture 权重。
    maximumWeights = max(posteriorParameters(i).w, [], 2);
    offsetWeights = posteriorParameters(i).w - maximumWeights;
    posteriorParameters(i).w = exp(offsetWeights) ./ sum(exp(offsetWeights), 2);
end
%% 3. 输出三个关联后端共用但字段不同的矩阵表示
% LBP association matrices
associationMatrices.Psi = L ./ eta;
associationMatrices.phi = phi;
associationMatrices.eta = eta;
% Gibbs sampler association matrices
associationMatrices.P = L./ (L + eta);
associationMatrices.L = [eta L];
associationMatrices.R = [(phi ./ eta) ones(numberOfObjects, numberOfMeasurements)];
% Murty's algorithm association matrices
associationMatrices.C = -log(L);
end
