function stateEstimates = runIcLmbFilter(model, measurements, sensorTrajectories)
% RUNICLMBFILTER -- Run the iterated-corrector LMB (IC-LMB) filter for a given simulated scenario.
%   stateEstimates = runIcLmbFilter(model, measurements)
%
%   Determine the objects' state estimates using the IC-LMB filter.
%   文件导读：
%       IC-LMB 顺序更新 baseline。它不像 runParallelUpdateLmbFilter 那样先
%       得到每个传感器的 local posterior 再融合，而是在同一时刻内按传感器
%       顺序连续更新同一个 LMB posterior，用作 iterated-corrector 对照。
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

%% 1. 初始化 LMB component 和输出容器
simulationLength = length(measurements);
% Struct containing objects' Bernoulli parameters and metadata
objects = model.object;

% 移动传感器轨迹会传给 per-sensor association matrix 构造器。
if nargin >= 3 && ~isempty(sensorTrajectories)
    model.sensorTrajectories = sensorTrajectories;
end
% Output struct
stateEstimates.labels = cell(simulationLength, 1);
stateEstimates.mu = cell(simulationLength, 1);
stateEstimates.Sigma = cell(simulationLength, 1);
stateEstimates.objects = objects;
%% 2. 时间递推：prediction 后按传感器顺序逐次 correction
for t = 1:simulationLength
    %% 2.1 预测
    objects = lmbPredictionStep(objects, model, t);
    %% 2.2 按传感器顺序进行 measurement update
    for s = 1:model.numberOfSensors
        if (numel(measurements{s, t}))
            % 移动传感器场景下，关联矩阵需要当前时刻来读取传感器位置。
            if model.sensorMotionEnabled
                [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, measurements{s, t}, model, s, t);
            else
                [associationMatrices, posteriorParameters] = generateLmbSensorAssociationMatrices(objects, measurements{s, t}, model, s);
            end
            if (strcmp(model.dataAssociationMethod, 'LBP'))
                % Data association by way of loopy belief propagation
                [r, W] = loopyBeliefPropagation(associationMatrices, model.lbpConvergenceTolerance, model.maximumNumberOfLbpIterations);
            elseif(strcmp(model.dataAssociationMethod, 'Gibbs'))
                % Data association by way of Gibbs sampling
                [r, W] = lmbGibbsSampling(associationMatrices, model.numberOfSamples);
            else
                % Data association by way of Murty's algorithm
                [r, W] = lmbMurtysAlgorithm(associationMatrices, model.numberOfAssignments);
            end
            % IC 路径直接把该传感器 posterior 写回 objects，供下一个传感器继续更新。
            objects = computePosteriorLmbSpatialDistributions(objects, r, W, posteriorParameters, model);
        else
            % 当前传感器没有测量时，只做该传感器的 missed-detection 更新。
            for i = 1:numel(objects)
                objects(i).r = (objects(i).r * ( 1- model.detectionProbability(s))) / (1 - objects(i).r * model.detectionProbability(s));
            end
        end
    end
    %% 2.3 轨迹剪枝：低存在概率 component 不再参与后续递推
    objectsLikelyToExist = [objects.r] > model.existenceThreshold;
    % 已形成足够长轨迹的低概率 component 仍保存到输出列表。
    discardedObjects = objects(~objectsLikelyToExist & ([objects.trajectoryLength] > model.minimumTrajectoryLength));
    stateEstimates.objects(end+1:end+numel(discardedObjects)) =  discardedObjects;
    objects = objects(objectsLikelyToExist);
    %% 2.4 MAP 基数/状态抽取
    [nMap, mapIndices] = lmbMapCardinalityEstimate([objects.r]);
    % 写出当前时刻的 RFS estimate。
    stateEstimates.labels{t} = zeros(2, nMap);
    stateEstimates.mu{t} = cell(1, nMap);
    stateEstimates.Sigma{t} = cell(1, nMap);
    for i = 1:nMap
        j = mapIndices(i);
        % Gaussians in the posterior GM are sorted according to weight
        stateEstimates.labels{t}(:, i) = [objects(j).birthTime; objects(j).birthLocation];
        stateEstimates.mu{t}{i} = objects(j).mu{1};
        stateEstimates.Sigma{t}{i} = objects(j).Sigma{1};
    end
    %% 2.5 更新轨迹缓存
    for i = 1:numel(objects)
        j = objects(i).trajectoryLength;
        objects(i).trajectoryLength = j + 1;
        objects(i).trajectory(:, j+1) = objects(i).mu{1};
        objects(i).timestamps(j+1) = t;
    end 
end
%% 3. 收尾：保存仍存活且足够长的轨迹
discardedObjects = objects(([objects.trajectoryLength] > model.minimumTrajectoryLength));
numberOfDiscardedObjects = numel(discardedObjects);
stateEstimates.objects(end+1:end+numberOfDiscardedObjects) =  discardedObjects;
end
