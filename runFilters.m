% RUNFILTERS - Run the single-sensor LMB or LMBM filters
%
% 文件导读：
%   单传感器最小可运行示例。阅读时按“建模 -> 生成观测 -> 运行滤波器
%   -> 绘图”的顺序看即可。它适合用来确认基础 LMB/LMBM 流程能跑通，
%   再继续看多传感器和动态权重实验。

%% Admin
close all; clc;
setPath;
%% 1. 构造单传感器模型
useLmbFilter = true; % Use LMB filter, or use LMBM filter
model = generateModel(10, 0.95, 'LBP', 'Fixed');
%% 2. 生成 ground truth 和观测
[groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model);
%% 3. 根据开关选择 LMB 或 LMBM 主循环
if (useLmbFilter)
    stateEstimates = runLmbFilter(model, measurements);
else
    stateEstimates = runLmbmFilter(model, measurements);
end
%% 4. 可视化估计结果和真值
plotResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs);
