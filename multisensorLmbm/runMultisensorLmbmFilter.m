function stateEstimates = runMultisensorLmbmFilter(model, measurements)
% RUNMULTISENSORLMBMFILTER -- Run the multi-sensor LMBM filter for a given simulated scenario.
%   stateEstimates = runMultisensorLmbmFilter(model, measurements)
%
%   Determine the objects' state estimates using the multi-sensor LMBM filter.
%   WARNING: This filter is impossibly slow, and very memory intensive.
%   If you use too many objects and sensors, then it is likely to exceed 
%   Matlab's memory limit and throw an error.
%   文件导读：
%       多传感器 LMBM 参考实现。它属于核心算法族，但多传感器关联空间
%       增长很快，实际主要用于小规模 baseline 或理论对照。
%
%   See also generateMultisensorModel, generateMultisensorGroundTruth 
%
%   Inputs
%       model - struct. A struct with the fields declared in generateModel.
%       measurements - cell array. An array containing the measurements for
%           each time-step of the simulation. See also generateModel.
%
%   Output
%       stateEstimates - struct. A struct containing the LMB filter's
%           approximate MAP estimate for each time-step of the simulation, as
%           well as the objects' trajectories.

%% 1. 初始化 hypotheses、trajectory 缓存和输出结构
simulationLength = length(measurements);
% Struct containing objects' Bernoulli parameters and metadata
hypotheses = model.hypotheses;
objects = model.trajectory;
% Output struct
stateEstimates.labels = cell(simulationLength, 1);
stateEstimates.mu = cell(simulationLength, 1);
stateEstimates.Sigma = cell(simulationLength, 1);
stateEstimates.objects = objects;
%% 2. 时间递推：多传感器关联会让 hypothesis 分支快速增长
for t = 1:simulationLength
    %% 2.1 添加当前时刻 birth trajectory 占位
    [model.birthTrajectory.birthTime] = deal(t);
    objects(end+1:end+model.numberOfBirthLocations) = model.birthTrajectory; 
    %% 2.2 预分配 posterior hypothesis 列表，并检查是否存在任何传感器量测
    posteriorHypotheses = repmat(model.hypotheses, 0, 1);
    % 只要有一个传感器有量测，就进入多传感器关联更新。
    measurementsAreAvailable = false;
    for s = 1:model.numberOfSensors
        measurementsAreAvailable = measurementsAreAvailable || (numel(measurements{s, t}) > 0);
    end
    %% 2.3 对每个 prior hypothesis 做 prediction 和 multi-sensor update
    for i = 1:numel(hypotheses)
        %% 2.3.1 hypothesis prediction
        priorHypothesis = lmbmPredictionStep(hypotheses(i), model, t);
        %% 2.3.2 multi-sensor measurement update
        if (measurementsAreAvailable)
            % 生成高维多传感器关联 likelihood 表。
            [L, posteriorParameters] = generateMultisensorLmbmAssociationMatrices(priorHypothesis, measurements(:, t), model);
            % 用多传感器 Gibbs sampler 生成关联事件。
            A = multisensorLmbmGibbsSampling(L, model.numberOfSamples);
            % 将关联事件重建为 posterior hypotheses。
            newHypotheses = determineMultisensorPosteriorHypothesisParameters(A, L, posteriorParameters, priorHypothesis);
            % 追加到 posterior hypothesis 列表。
            posteriorHypotheses(end+1:end+numel(newHypotheses)) = newHypotheses;
        else
            priorHypothesis.r = (prod(1-model.detectionProbability) * priorHypothesis.r) ./ (1 - priorHypothesis.r + prod(1-model.detectionProbability) * priorHypothesis.r);
            posteriorHypotheses(end+1) = priorHypothesis; 
        end
    end
    %% 2.4 归一化和剪枝 hypotheses
    [hypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model);
    %% 2.5 剪枝 trajectory，长轨迹仍保留输出
    discardedObjects = objects(~objectsLikelyToExist' & ([objects.trajectoryLength] > model.minimumTrajectoryLength));
    stateEstimates.objects(end+1:end+numel(discardedObjects)) =  discardedObjects;
    % Keep objects with high existence probabilities
    objects = objects(objectsLikelyToExist);
    %% 2.6 从最高权重 hypothesis 中抽取当前时刻状态估计
    [cardinalityEstimate, extractionIndices] = lmbmStateExtraction(hypotheses, false);
    % 写出 RFS estimate。
    stateEstimates.labels{t} = zeros(2, cardinalityEstimate);
    stateEstimates.mu{t} = cell(1, cardinalityEstimate);
    stateEstimates.Sigma{t} = cell(1, cardinalityEstimate);
    for i = 1:cardinalityEstimate
        j = extractionIndices(i);
        % hypotheses 已按权重排序，取第一个 hypothesis 的 component。
        stateEstimates.labels{t}(:, i) = [hypotheses(1).birthTime(j); hypotheses(1).birthLocation(j)];
        stateEstimates.mu{t}{i} = hypotheses(1).mu{j};
        stateEstimates.Sigma{t}{i} = hypotheses(1).Sigma{j};
    end
    %% 2.7 更新 trajectory 缓存
    for i = 1:numel(objects)
        j = objects(i).trajectoryLength;
        objects(i).trajectoryLength = j + 1;
        objects(i).trajectory(:, j+1) = hypotheses(1).mu{i};
        objects(i).timestamps(j+1) = t;
    end 
end
%% 3. 收尾：保存仍存活且足够长的轨迹
discardedObjects = objects(([objects.trajectoryLength] > model.minimumTrajectoryLength));
numberOfDiscardedObjects = numel(discardedObjects);
stateEstimates.objects(end+1:end+numberOfDiscardedObjects) =  discardedObjects;
end
