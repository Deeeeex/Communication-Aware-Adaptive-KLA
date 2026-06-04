% RUNMULTISENSORFILTERS - Run the multi-sensor LMB or LMBM filters
%
% 文件导读：
%   集中式多传感器最小示例。这个脚本展示完整主线：
%   传感器模型 -> 多传感器观测 -> 可选通信约束 -> IC/PU/GA/AA/LMBM
%   分支 -> 绘图。更复杂的 formation 和论文实验脚本沿用同一条流程。

%% Admin
close all; clc;
setPath;
%% 1. 选择多传感器滤波/融合分支
filterType = 'PU'; % 'IC', 'PU', 'LMBM'
%% 2. 配置传感器数量、检测率、杂波和测量噪声
numberOfSensors = 3;
clutterRates = [5 5 5];
detectionProbabilities = [0.67 0.70 0.73];
q = [4 3 2];
model = generateMultisensorModel(numberOfSensors, clutterRates, detectionProbabilities, q, 'GA', 'LBP', 'Fixed');
%% 3. 动态融合权重开关；此示例默认关闭，只保留配置格式
model.adaptiveFusion = struct();
model.adaptiveFusion.enabled = false;
model.adaptiveFusion.emaAlpha = 0.7;
model.adaptiveFusion.minWeight = 0.05;
%% 4. 生成多传感器观测
[groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model);
%% 5. 可选通信约束：带宽、链路丢包、节点 outage
commConfig = struct();
commConfig.level = 3; % 0=ideal, 1=bandwidth, 2=link loss, 3=node outage
commConfig.globalMaxMeasurementsPerStep = 25;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.maxOutageNodes = 1;
[measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
%% 6. 根据 filterType 调用对应滤波主循环
if(strcmp(filterType, 'IC'))
    % Iterated-corrector LMB (IC-LMB) filter
    stateEstimates = runIcLmbFilter(model, measurementsDelivered);
elseif(strcmp(filterType, 'PU'))
    % Parallel measurement update: PU-, GA-, or AA-LMB filters
    stateEstimates = runParallelUpdateLmbFilter(model, measurementsDelivered, commStats);
else
    % Multisensor LMBM filter
    stateEstimates = runMultisensorLmbmFilter(model, measurementsDelivered);
end
%% 7. 绘制多传感器跟踪结果
plotMultisensorResults(model, measurementsDelivered, groundTruth, stateEstimates, groundTruthRfs);
 
