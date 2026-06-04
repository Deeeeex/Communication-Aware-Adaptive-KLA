function [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, z, model, s, currentTime)
% GENERATELMBSENSORASSOCIATIONMATRICES -- Compute the association matrices required for
%    [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, z, model, s)
%    [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, z, model, s, currentTime)
%
%   This function computes the association matrices required by the LBP,
%   Gibbs sampler, and Murty's algorithms for a given sensor. It also 
%   determines the measurement-updated components requried for each sensor's measurement update.
%   Supports mobile sensors with position-dependent measurements.
%   文件导读：
%       每个传感器的关联矩阵构造器，是单传感器 generateLmbAssociationMatrices
%       的多传感器版本。它额外处理 sensor-indexed C/Q、移动传感器几何、
%       state-dependent p_D/Q、NIS 诊断和 association ambiguity 诊断；这些
%       诊断可被动态权重模块读取。
%
%   See also runLmbFilter, generateMultisensorModel, loopyBeliefPropagation, lmbGibbsSampling, lmbMurtysAlgorithm
%
%   Inputs
%       objects - struct. A struct containing the prior LMB's Bernoulli components.
%       z - cell array. A cell array of measurements for the
%           current time-step.
%       model - struct. A struct with the fields declared in generateMultisensorModel.
%       s - integer. The sensor number.
%       currentTime - integer. The current time step (optional, for mobile sensors).
%
%   Output
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       posteriorParameters - struct. A struct whose fields are an object's
%           posterior spatial distribution parameters.
%
%   This function computes the association matrices required by the LBP,
%   Gibbs sampler, and Murty's algorithms for a given sensor. It also 
%   determines the measurement-updated components requried for each sensor's measurement update.
%
%   See also runLmbFilter, generateMultisensorModel, loopyBeliefPropagation, lmbGibbsSampling, lmbMurtysAlgorithm
%
%   Inputs
%       objects - struct. A struct containing the prior LMB's Bernoulli components.
%       z - cell array. A cell array of measurements for the
%           current time-step.
%       model - struct. A struct with the fields declared in generateMultisensorModel.
%       s - integer. The sensor number.
%
%   Output
%       associationMatrices - struct. A struct whose fields are the arrays required 
%           by the various data association algorithms.
%       posteriorParameters - struct. A struct whose fields are an object's
%           posterior spatial distribution parameters.
%

%% 1. 分配关联矩阵、诊断量和 posterior component 容器
numberOfObjects = numel(objects);
numberOfMeasurements = numel(z);
if nargin < 5 || isempty(currentTime)
    currentTime = 1;
end
% Auxillary matrices
L = zeros(numberOfObjects, numberOfMeasurements);
phi = zeros(numberOfObjects, 1);
eta = zeros(numberOfObjects, 1);
nisMin = inf(1, numberOfMeasurements);
% Updated components for the objects' posterior spatial distributions
posteriorParameters.w = [];
posteriorParameters.mu = {};
posteriorParameters.Sigma = {};
posteriorParameters = repmat(posteriorParameters, 1, numberOfObjects);
%% 2. 对每个 Bernoulli component 计算该传感器下的漏检项和检测项
for i = 1:numberOfObjects
    % 第 1 行是 missed-detection component，后面每一行对应一条测量。
    posteriorParameters(i).w = -inf(numberOfMeasurements + 1, objects(i).numberOfGmComponents);
    posteriorParameters(i).mu  = repmat(objects(i).mu, numberOfMeasurements + 1, 1);
    posteriorParameters(i).Sigma = repmat(objects(i).Sigma, numberOfMeasurements + 1, 1);
    missedDetectionLikelihood = 0;
    %% 2.1 遍历该 Bernoulli 的 Gaussian mixture component
    for j = 1:objects(i).numberOfGmComponents
        % 移动传感器场景下，观测均值要根据当前传感器位置计算。
        if model.sensorMotionEnabled && nargin >= 4
            sensorPos = model.sensorTrajectories{s}(1:2, currentTime);
            targetPos = objects(i).mu{j}(1:2);
            muZ = sensorPos + model.C{s} * [targetPos - sensorPos; 0; 0];
        else
            muZ = model.C{s} * objects(i).mu{j};
        end

        % p_D 和 Q 可以随目标-传感器几何变化；这是动态权重质量诊断的输入之一。
        [pdSensor, qSensor] = evaluateSensorQuality(model, s, objects(i).mu{j}, currentTime);
        pdSensorForLog = max(pdSensor, realmin);
        missedDetectionForLog = max(1 - pdSensor, realmin);
        posteriorParameters(i).w(1, j) = log(objects(i).w(j)) + log(missedDetectionForLog);
        missedDetectionLikelihood = missedDetectionLikelihood + ...
            objects(i).w(j) * (1 - pdSensor);

        Z = model.C{s} * objects(i).Sigma{j} * model.C{s}' + qSensor;
        logGaussianNormalisingConstant = - (0.5 * model.zDimension) * log(2 * pi) - 0.5 * log(det(Z));
        logLikelihoodRatioTerms = log(objects(i).r) + log(pdSensorForLog) + log(objects(i).w(j)) - log(model.clutterPerUnitVolume(s));
        projectionCov = objects(i).Sigma{j} * model.C{s}';
        K = projectionCov / Z;
        SigmaUpdated = (eye(model.xDimension) - K * model.C{s}) * objects(i).Sigma{j};
        % 将当前 Gaussian component 对所有测量的贡献累加到 L 和 posteriorParameters。
        for k = 1:numberOfMeasurements
            % L(i,k) 是 object i 由当前传感器 measurement k 生成的 likelihood ratio。
            nu = z{k} - muZ;
            nisValue = nu' * (Z \ nu);
            gaussianLogLikelihood = logGaussianNormalisingConstant - 0.5 * nisValue;
            L(i, k) = L(i, k) + exp(logLikelihoodRatioTerms + gaussianLogLikelihood);
            if nisValue < nisMin(k)
                nisMin(k) = nisValue;
            end
            % 保存当前测量条件下的 posterior Gaussian component。
            posteriorParameters(i).w(k+1, j) = log(objects(i).w(j)) + gaussianLogLikelihood + log(pdSensorForLog) - log(model.clutterPerUnitVolume(s));
            posteriorParameters(i).mu{k+1, j} = objects(i).mu{j} + K * nu;
            posteriorParameters(i).Sigma{k+1, j} = SigmaUpdated;
        end
    end
    % Populate auxiliary LBP parameters. For state-dependent p_D, this is
    % the mixture expectation of the missed-detection likelihood.
    phi(i) = missedDetectionLikelihood * objects(i).r;
    eta(i) = 1 - objects(i).r + phi(i);
    % Normalise weights
    maximumWeights = max(posteriorParameters(i).w, [], 2);
    offsetWeights = posteriorParameters(i).w - maximumWeights;
    posteriorParameters(i).w = exp(offsetWeights) ./ sum(exp(offsetWeights), 2);
end
%% Output association matrices
associationMatrices.r = reshape([objects.r], numberOfObjects, 1);
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
% Association ambiguity / confidence summary
associationMatrices.associationConfidence = computeAssociationConfidence(L);
associationMatrices.associationAmbiguityScore = associationMatrices.associationConfidence;
% Innovation consistency (NIS-based)
if numberOfMeasurements > 0
    cfg = struct();
    if isfield(model, 'adaptiveFusion') && isstruct(model.adaptiveFusion)
        cfg = model.adaptiveFusion;
    end
    validNis = nisMin(isfinite(nisMin));
    if isempty(validNis)
        associationMatrices.innovationScore = 1;
        associationMatrices.innovationPenalty = 1;
        associationMatrices.nisAgg = NaN;
        associationMatrices.nisNorm = NaN;
        associationMatrices.nisLowerBound = NaN;
        associationMatrices.nisUpperBound = NaN;
        associationMatrices.nisDeviation = NaN;
        return;
    else
        useRobust = false;
        if isfield(cfg, 'robustNIS') && cfg.robustNIS && ...
                isfield(model, 'lmbParallelUpdateMode') && strcmpi(model.lmbParallelUpdateMode, 'GA')
            useRobust = true;
        end
        useQuantile = getConfigField(cfg, 'nisQuantileEnabled', true);
        quantileValue = getConfigField(cfg, 'nisQuantile', 0.7);
        if useRobust && useQuantile
            quantileValue = min(max(quantileValue, 0), 1);
            nisAgg = computeQuantile1d(validNis, quantileValue);
        elseif useRobust
            nisAgg = median(validNis);
        else
            nisAgg = mean(validNis);
        end
    end
    dof = max(1, model.zDimension);
    nisNorm = nisAgg / dof;
    confidenceLevel = getConfigField(cfg, 'nisConsistencyConfidence', 0.5);
    confidenceLevel = min(max(confidenceLevel, 1e-3), 0.999);
    lowerTail = 0.5 * (1 - confidenceLevel);
    upperTail = 1 - lowerTail;
    lowerBound = computeChiSquareQuantile(lowerTail, dof) / dof;
    upperBound = computeChiSquareQuantile(upperTail, dof) / dof;
    [lowerDeviation, upperDeviation] = computeLogIntervalDeviation(nisNorm, lowerBound, upperBound);
    lowerScale = getConfigField(cfg, 'nisPenaltyLowerScale', ...
        0.25 * getConfigField(cfg, 'nisPenaltyScale', 4.0));
    upperScale = getConfigField(cfg, 'nisPenaltyUpperScale', ...
        getConfigField(cfg, 'nisPenaltyScale', 4.0));
    lowerPower = getConfigField(cfg, 'nisPenaltyLowerPower', 2.0);
    upperPower = getConfigField(cfg, 'nisPenaltyUpperPower', 2.0);
    penaltyExponent = lowerScale * lowerDeviation^lowerPower + ...
        upperScale * upperDeviation^upperPower;
    penalty = exp(-penaltyExponent);
    minPenalty = 0.0;
    maxPenalty = getConfigField(cfg, 'nisPenaltyMax', 1.0);
    if useRobust
        minPenalty = getConfigField(cfg, 'nisPenaltyMin', ...
            getConfigField(cfg, 'robustNISMin', 0.2));
    end
    penalty = min(max(penalty, minPenalty), maxPenalty);
    associationMatrices.innovationScore = penalty;
    associationMatrices.innovationPenalty = penalty;
    associationMatrices.nisAgg = nisAgg;
    associationMatrices.nisNorm = nisNorm;
    associationMatrices.nisLowerBound = lowerBound;
    associationMatrices.nisUpperBound = upperBound;
    associationMatrices.nisDeviation = lowerDeviation + upperDeviation;
    associationMatrices.nisLowerDeviation = lowerDeviation;
    associationMatrices.nisUpperDeviation = upperDeviation;
else
    associationMatrices.innovationScore = 1;
    associationMatrices.innovationPenalty = 1;
end
end

function score = computeAssociationConfidence(L)
score = 1;
if isempty(L) || size(L, 2) == 0
    return;
end

measurementScores = zeros(1, size(L, 2));
scoreCount = 0;
for k = 1:size(L, 2)
    values = L(:, k);
    values = values(isfinite(values) & values > 0);
    if isempty(values)
        continue;
    end

    probs = values / sum(values);
    if numel(probs) <= 1
        measurementScore = 1;
    else
        entropyValue = -sum(probs .* log(max(probs, eps)));
        normalizedEntropy = entropyValue / log(numel(probs));
        sortedProbs = sort(probs, 'descend');
        topProb = sortedProbs(1);
        secondProb = sortedProbs(2);
        marginScore = (topProb - secondProb) / max(topProb, eps);
        measurementScore = 0.5 * (1 - normalizedEntropy) + 0.5 * marginScore;
    end

    scoreCount = scoreCount + 1;
    measurementScores(scoreCount) = min(max(measurementScore, 0), 1);
end

if scoreCount > 0
    score = mean(measurementScores(1:scoreCount));
end
end

function value = getConfigField(cfg, fieldName, defaultValue)
if isfield(cfg, fieldName)
    value = cfg.(fieldName);
else
    value = defaultValue;
end
end

function [lowerDeviation, upperDeviation] = computeLogIntervalDeviation(value, lowerBound, upperBound)
value = max(value, eps);
lowerBound = max(lowerBound, eps);
upperBound = max(upperBound, lowerBound + eps);
lowerDeviation = 0;
upperDeviation = 0;
if value < lowerBound
    lowerDeviation = log(lowerBound / value);
elseif value > upperBound
    upperDeviation = log(value / upperBound);
end
end

function value = computeChiSquareQuantile(p, dof)
p = min(max(p, 1e-12), 1 - 1e-12);
value = 2 * gammaincinv(p, dof / 2);
end

function value = computeQuantile1d(values, q)
sortedValues = sort(values(:));
numValues = numel(sortedValues);
if numValues == 0
    value = 0;
    return;
end
if numValues == 1
    value = sortedValues(1);
    return;
end
index = 1 + (numValues - 1) * q;
lowerIndex = floor(index);
upperIndex = ceil(index);
if lowerIndex == upperIndex
    value = sortedValues(lowerIndex);
else
    fraction = index - lowerIndex;
    value = sortedValues(lowerIndex) + fraction * ...
        (sortedValues(upperIndex) - sortedValues(lowerIndex));
end
end
