function stateEstimates = runLmbmFilter(model, measurements)
% RUNLMBMFILTER -- Run the LMBM filter for a given simulated scenario.
%   stateEstimates = runLmbmFilter(model, measurements)
%
%   Determine the objects' state estimates using the LMBM filter.
%   文件导读：
%       单传感器 LMBM 主循环。它不会像 LMB 那样每步折叠成一个 posterior
%       LMB，而是保留多个 global hypotheses。因此它更适合作为
%       hypothesis-managed 参考实现或小规模对照，计算量明显高于 runLmbFilter。
%
%   See also generateModel, generateGroundTruth
%
%   Inputs
%       model - struct. A struct with the fields declared in generateModel.
%       measurements - cell array. An array containing the measurements for
%           each time-step of the simulation. See also generateGroundTruth.
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
%% 2. 时间递推：每个 prior hypothesis 会分裂出多个 posterior hypotheses
for t = 1:simulationLength
    %% 2.1 添加当前时刻的 birth trajectory 占位
    [model.birthTrajectory.birthTime] = deal(t);
    objects(end+1:end+model.numberOfBirthLocations) = model.birthTrajectory; 
    %% 2.2 预分配 posterior hypothesis 列表
    posteriorHypotheses = repmat(model.hypotheses, 0, 1);
    %% 2.3 对每个 prior hypothesis 做 prediction 和 measurement update
    for i = 1:numel(hypotheses)
        %% 2.3.1 hypothesis prediction
        priorHypothesis = lmbmPredictionStep(hypotheses(i), model, t);
        %% 2.3.2 measurement update：有量测时按关联事件生成新 hypotheses
        if (numel(measurements{t}))
            % 先生成 LMBM 关联矩阵和每个关联事件对应的 posterior 参数。
            [associationMatrices, posteriorParameters] = generateLmbmAssociationMatrices(priorHypothesis, measurements{t}, model);
            % 根据配置选择 Murty 或 Gibbs 生成关联事件。
            if(strcmp(model.dataAssociationMethod, 'Murty'))
                V = murtysAlgorithmWrapper(associationMatrices.C, model.numberOfAssignments);
            else
                V = lmbmGibbsSampling(associationMatrices.P, associationMatrices.C, model.numberOfSamples);
            end
            % 将关联事件转换为 posterior hypothesis。
            newHypotheses = determinePosteriorHypothesisParameters(V, associationMatrices.L, posteriorParameters, priorHypothesis);
            % 追加到全局 posterior hypothesis 列表。
            posteriorHypotheses(end+1:end+numel(newHypotheses)) = newHypotheses;
        else
            priorHypothesis.r = ((1-model.detectionProbability) * priorHypothesis.r) ./ (1 - model.detectionProbability * priorHypothesis.r);
            posteriorHypotheses(end+1) = priorHypothesis; 
        end
    end
    %% 2.4 归一化 hypothesis 权重，并剪掉低权重 hypotheses / 低 r components
    [hypotheses, objectsLikelyToExist] = lmbmNormalisationAndGating(posteriorHypotheses, model);
    %% 2.5 根据 component mask 剪枝 trajectory，长轨迹仍保留输出
    discardedObjects = objects(~objectsLikelyToExist' & ([objects.trajectoryLength] > model.minimumTrajectoryLength));
    stateEstimates.objects(end+1:end+numel(discardedObjects)) =  discardedObjects;
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
