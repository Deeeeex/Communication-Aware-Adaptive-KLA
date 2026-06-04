function stateEstimates = runLmbFilter(model, measurements)
% RUNLMBFILTER -- Run the LMB filter for a given simulated scenario.
%   stateEstimates = runLmbFilter(model, measurements)
%
%   Determine the objects' state estimates using the LMB filter.
%   文件导读：
%       单传感器 LMB 主循环。每个时刻都按固定顺序推进：
%       预测 -> 构造关联矩阵 -> 选择 LBP/Gibbs/Murty 求关联边缘概率
%       -> 重组 posterior Gaussian mixture -> 剪枝 -> MAP 基数/状态输出
%       -> 轨迹记录。阅读这个文件可以把握 LMB 主算法的骨架。
%
%   See also generateModel, generateGroundTruth, lmbPredictionStep,
%   generateLmbAssociationMatrices, loopyBeliefPropagation, lmbGibbsSampling, 
%   lmbMurtysAlgorithm, computePosteriorLmbSpatialDistributions, lmbMapCardinalityEstimate
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

%% 1. 初始化：objects 是当前 LMB 的 Bernoulli component 集合
simulationLength = length(measurements);
% Struct containing objects' Bernoulli parameters and metadata
objects = model.object;
% Output struct
stateEstimates.labels = cell(simulationLength, 1);
stateEstimates.mu = cell(simulationLength, 1);
stateEstimates.Sigma = cell(simulationLength, 1);
stateEstimates.objects = objects;
%% 2. 时间递推：每个时刻执行 prediction + measurement update + extraction
for t = 1:simulationLength
    %% 2.1 预测：传播已有 Bernoulli，并加入当前时刻的 birth components
    objects = lmbPredictionStep(objects, model, t);
    %% 2.2 量测更新：有观测时走关联求解，无观测时只做 missed-detection 更新
    if (numel(measurements{t}))
        % 先把 likelihood、missed detection 和 Kalman 更新组件统一整理好。
        [associationMatrices, posteriorParameters] = generateLmbAssociationMatrices(objects, measurements{t}, model);
        if (strcmp(model.dataAssociationMethod, 'LBP'))
            % LBP：默认快速近似关联后端。
            [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);
        elseif(strcmp(model.dataAssociationMethod, 'LBPFixed'))
            [r, W] = fixedLoopyBeliefPropagation(associationMatrices, model.maximumNumberOfLbpIterations);
        elseif(strcmp(model.dataAssociationMethod, 'Gibbs'))
            % Gibbs：采样式近似后端，用于对照或不确定性分析。
            [r, W] = lmbGibbsSampling(associationMatrices, model.numberOfSamples);
        else
            % Murty：K-best assignment 后端，通常更贵，适合小规模基准。
            [r, W] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);
        end
        % 将关联边缘概率 W 写回 Gaussian mixture，得到 posterior LMB。
        objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
    else
        % 没有量测时，每个 Bernoulli 只根据漏检概率更新存在概率。
        for i = 1:numel(objects)
            objects(i).r = (objects(i).r * (1-model.detectionProbability)) / (1 - objects(i).r * model.detectionProbability);
        end
    end
    %% 2.3 轨迹剪枝：低存在概率的 component 不再参与后续递推
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    % 如果低概率 component 已经形成足够长轨迹，仍保存到输出对象列表中。
    discardedObjects = objects(~objectsLikelyToExist & ([objects.trajectoryLength] > model.minimumTrajectoryLength));
    stateEstimates.objects(end+1:end+numel(discardedObjects)) =  discardedObjects;
    objects = objects(objectsLikelyToExist);
    %% 2.4 MAP 基数抽取：决定本时刻输出几个目标
    [nMap, mapIndices] = lmbMapCardinalityEstimate([objects.r]);
    % 按存在概率排序后的 component 输出为 RFS state estimate。
    stateEstimates.labels{t} = zeros(2, nMap);
    stateEstimates.mu{t} = cell(1, nMap);
    stateEstimates.Sigma{t} = cell(1, nMap);
    for i = 1:nMap
        j = mapIndices(i);
        % posterior GM 已按权重排序，取每个 component 的主 Gaussian。
        stateEstimates.labels{t}(:, i) = [objects(j).birthTime; objects(j).birthLocation];
        stateEstimates.mu{t}{i} = objects(j).mu{1};
        stateEstimates.Sigma{t}{i} = objects(j).Sigma{1};
    end
    %% 2.5 更新轨迹缓存：供可视化和长轨迹导出使用
    for i = 1:numel(objects)
        j = objects(i).trajectoryLength;
        objects(i).trajectoryLength = j + 1;
        objects(i).trajectory(:, j+1) = objects(i).mu{1};
        objects(i).timestamps(j+1) = t;
    end 
end
%% 3. 收尾：把仍存活且足够长的轨迹也写入输出
discardedObjects = objects(([objects.trajectoryLength] > model.minimumTrajectoryLength));
numberOfDiscardedObjects = numel(discardedObjects);
stateEstimates.objects(end+1:end+numberOfDiscardedObjects) =  discardedObjects;
end
