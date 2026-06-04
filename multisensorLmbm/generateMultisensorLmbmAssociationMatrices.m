function [L, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(hypothesis, z, model)
% GENERATEMULTISENSORLMBMASSOCIATIONMATRICES -- Compute the association matrices required by the data association algorithms.
%   [associationMatrices, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(hypothesis, z, model)
%
%   This function computes the association matrices required by multi-sensor
%   Gibbs sampler. It also determines the measurement-updated components that
%   are used to determine each object's posterior spatial distribution.
%   The output matrix L can be prohibitively large for a large number of 
%   objects and sensors. It can exceed Matlab's memory limit, and in
%   general this algorithm is very slow.
%   文件导读：
%       多传感器 LMBM 的关联 likelihood 构造器。它枚举每个 object 在多个
%       传感器上的联合关联事件，输出给 multi-sensor Gibbs sampler 和后续
%       posterior hypothesis 重建。该矩阵可能非常大。
%
%   See also runMultisensorLmbmFilter, generateMultisensorModel, multisensorLmbmGibbsSampling,
%
%   Inputs
%       hypothesis - struct. A struct containing the prior LMBM hypothesis' Bernoulli components.
%       z - cell array. A cell array of measurements for the
%           current time-step.
%       model - struct. A struct with the fields declared in generateModel.
%
%   Output
%       L - (m1 + 1, m2 + 1, ..., ms + 1, n) array. A log likelihood
%           matrix used for Gibbs sampling and evaluating the probability
%           density of an association event.
%       posteriorParameters - struct. A struct whose fields are a
%           hypothesis posterior distribution parameters.

%% 1. 根据每个传感器测量数确定高维关联矩阵大小
dimensions = [zeros(1, model.numberOfSensors) numel(hypothesis.r)];
for s = 1:model.numberOfSensors
    dimensions(s) = numel(z{s}) + 1;
end
numberOfEntries = prod(dimensions);
pageSizes = [1 cumprod(dimensions(1:end-1))];
%% 2. 预分配 likelihood 和 posterior 参数容器
% Log likelihood matrix for sampling
L = zeros(dimensions);
% Posterior Bernoulli parameters
r = zeros(dimensions);
mu = cell(dimensions);
Sigma = cell(dimensions);
%% 3. 枚举高维关联事件，逐项计算 likelihood ratio 和 posterior 参数
for ell = 1:numberOfEntries
    %% 3.1 将线性索引转换为多传感器关联向量
    u = convertFromLinearToCartesian(ell, pageSizes);
    i = u(end);
    a = u(1:end-1) - 1;
    %% 3.2 计算该联合关联事件的 log likelihood 和 posterior 参数
    [L(ell), r(ell), mu{ell}, Sigma{ell}] = determineLogLikelihoodRatio(i, a, z, hypothesis, model);
end
%% 4. 输出 posterior 参数张量
posteriorParameters.r = r;
posteriorParameters.mu = mu;
posteriorParameters.Sigma = Sigma;
end
%% 将线性索引转换为笛卡尔坐标形式的关联事件
function u = convertFromLinearToCartesian(ell, d)
%% 1. 初始化坐标容器
m = numel(d);
u = zeros(1, m);
n = ell;
%% 2. 逐维回退线性索引
for i = 1:m
    j = m - i + 1;
    zeta = floor(n / d(j));
    eta = mod(n, d(j));
    u(j) = zeta + (eta ~= 0);
    n = n - d(j) * (zeta - (eta == 0));
end
end
%% 计算一个 object 在多传感器联合关联事件下的 likelihood ratio
function [L, r, mu, Sigma] = determineLogLikelihoodRatio(i, a, measurements, hypothesis, model)
%% 1. 解析哪些传感器给该 object 分配了测量
assignments = a > 0;
numberOfAssignments = sum(assignments);
if (numberOfAssignments > 0)
    %% 2. 堆叠多传感器测量向量和观测矩阵
    z = zeros(model.zDimension * numberOfAssignments, 1);
    C = zeros(model.zDimension * numberOfAssignments, model.xDimension);
    counter = 0;
    for s = 1:model.numberOfSensors
        if (assignments(s))
            start = model.zDimension * counter + 1;
            finish = start + model.zDimension - 1;
            z(start:finish) = measurements{s}{a(s)};
            C(start:finish, :) = model.C{s};
            counter = counter + 1;
        end
    end
    %% 3. 计算联合 likelihood ratio 和 Kalman update
    % Noise matrix
    Q = blkdiag(model.Q{assignments});
    % Likelihood
    nu = z - C * hypothesis.mu{i};
    Z = C * hypothesis.Sigma{i} * C' + Q;
    ZInv = inv(Z);
    K = hypothesis.Sigma{i} * C' * ZInv;
    eta = -0.5 * log(det(2*pi*Z));
    Pd = sum(log(model.detectionProbability(assignments))) + sum(log(1 - model.detectionProbability(~assignments)));
    kappa = sum(log(model.clutterPerUnitVolume(assignments)));
    L = log(hypothesis.r(i)) + Pd + eta - (0.5 * nu' * ZInv * nu)  - kappa;
    %% 4. 写出检测事件下的 posterior 参数
    r = 1;
    mu = hypothesis.mu{i} + K * nu;
    Sigma = (eye(model.xDimension) - K * C) * hypothesis.Sigma{i};
else
    numerator = hypothesis.r(i) * prod(1 - model.detectionProbability);
    denominator =  1 - hypothesis.r(i) + numerator;
    % Output
    L = log(denominator);
    r = numerator / denominator;
    mu = hypothesis.mu{i};
    Sigma = hypothesis.Sigma{i};
end
end
