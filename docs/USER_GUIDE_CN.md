# 使用文档（中文）

本项目提供多目标跟踪的 LMB / LMBM 及多传感器扩展实现，以下文档聚焦“怎么跑起来 + 关键参数在哪改 + 对应代码位置”，方便你按需定制。

## 1. 快速开始

### 1.1 单传感器最小示例

最直接的入口脚本是 `runFilters.m`，内部流程依次调用模型生成、数据生成、滤波与绘图（见 `runFilters.m`）。

```matlab
setPath; % 加载路径，见 setPath.m
model = generateModel(10, 0.95, 'LBP', 'Fixed'); % common/generateModel.m
[groundTruth, measurements, groundTruthRfs] = generateGroundTruth(model); % common/generateGroundTruth.m
stateEstimates = runLmbFilter(model, measurements); % lmb/runLmbFilter.m
plotResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs); % common/plotResults.m
```

要切换为 LMBM 版本，把 `runLmbFilter` 换成 `runLmbmFilter`（见 `lmbm/runLmbmFilter.m`），但注意 LMBM 不支持 LBP。

### 1.2 多传感器最小示例

入口脚本 `runMultisensorFilters.m` 展示了 IC / PU / LMBM 的切换逻辑（见 `runMultisensorFilters.m`）。

```matlab
setPath;
model = generateMultisensorModel(3, [5 5 5], [0.67 0.70 0.73], [4 3 2], ...
    'PU', 'LBP', 'Fixed'); % common/generateMultisensorModel.m
[groundTruth, measurements, groundTruthRfs] = generateMultisensorGroundTruth(model); % common/generateMultisensorGroundTruth.m
stateEstimates = runParallelUpdateLmbFilter(model, measurements); % multisensorLmb/runParallelUpdateLmbFilter.m
plotMultisensorResults(model, measurements, groundTruth, stateEstimates, groundTruthRfs); % common/plotMultisensorResults.m
```

要用 IC-LMB，调用 `multisensorLmb/runIcLmbFilter.m`；要用多传感器 LMBM，调用 `multisensorLmbm/runMultisensorLmbmFilter.m`（注意该实现非常慢且占用内存，源码里有警告）。

## 2. 路径与入口脚本

- 路径初始化：`setPath.m`（将 `common/`、`lmb/`、`lmbm/`、`multisensorLmb/`、`multisensorLmbm/` 等加入 MATLAB path）
- 单传感器 demo：`runFilters.m`
- 多传感器 demo：`runMultisensorFilters.m`
- 综合性能分析：`runAdvancedPerformanceAnalysis.m`

### 通信层仿真（可选）

多传感器场景可在滤波前插入通信层仿真，入口脚本 `runMultisensorFilters.m` 已提供 `commConfig` 示例配置。通信层由 `common/applyCommunicationModel.m` 实现，详细规则与参数说明见 `docs/COMMUNICATION_SIMULATION_CN.md`。

## 3. 模型与参数配置（重点字段与代码位置）

### 3.1 单传感器模型

模型由 `common/generateModel.m` 生成。常用参数包括：

- 基本参数：`clutterRate`、`detectionProbability`、`dataAssociationMethod`、`scenarioType`
- 数据关联方法：`model.dataAssociationMethod`（`LBP` / `Gibbs` / `Murty` / `LBPFixed`）
- LBP 参数：`model.maximumNumberOfLbpIterations`、`model.lbpConvergenceTolerance`
- Gibbs 参数：`model.numberOfSamples`
- Murty 参数：`model.numberOfAssignments`
- 轨迹门控：`model.existenceThreshold`、`model.minimumTrajectoryLength`
- 高斯混合截断：`model.gmWeightThreshold`、`model.maximumNumberOfGmComponents`

示例（见 `common/generateModel.m` 定义的字段）：

```matlab
model = generateModel(10, 0.95, 'LBP', 'Fixed');
model.maximumNumberOfLbpIterations = 500;
model.lbpConvergenceTolerance = 1e-5;
model.existenceThreshold = 1e-2;
```

### 3.2 多传感器模型

模型由 `common/generateMultisensorModel.m` 生成，额外包含传感器相关配置：

- 传感器数量：`numberOfSensors`
- 每个传感器的 `clutterRates` / `detectionProbabilities`
- 观测噪声：`q`（标量数组，内部生成 `Q{i} = q(i)^2 * I`）
- 融合模式：`model.lmbParallelUpdateMode`（`PU` / `GA` / `AA`）
- 传感器权重：`model.aaSensorWeights`、`model.gaSensorWeights`

示例（见 `common/generateMultisensorModel.m`）：

```matlab
model = generateMultisensorModel(3, [5 8 12], [0.9 0.85 0.95], [4 3 2], ...
    'PU', 'LBP', 'Fixed');
model.lmbParallelUpdateMode = 'GA';
model.gaSensorWeights = [0.4 0.3 0.3];
```

## 4. 数据格式约定（与代码一致）

这部分非常关键，绘图和滤波代码都假设了下述结构。

### 4.1 单传感器 measurements

`measurements` 是长度为 `T` 的 cell，每个 `measurements{t}` 也是 cell，内部每个元素是 `2x1` 观测向量（见 `common/generateGroundTruth.m` 和 `common/plotResults.m`）。

```matlab
measurements{t}{k} = [x; y]; % 2x1
```

### 4.2 多传感器 measurements

`measurements` 是 `sensors x time` 的 cell 矩阵，每个 `measurements{s, t}` 也是 cell，内部为 `2x1` 观测向量（见 `common/generateMultisensorGroundTruth.m` 和 `common/plotMultisensorResults.m`）。

```matlab
measurements{s, t}{k} = [x; y];
```

### 4.3 groundTruth / groundTruthRfs

生成函数 `common/generateGroundTruth.m` / `common/generateMultisensorGroundTruth.m` 返回：

- `groundTruth`：每个对象一条轨迹，矩阵列为 `[t; x; y; vx; vy]`。
- `groundTruthRfs`：RFS 形式的真值（字段 `x` / `mu` / `Sigma` / `cardinality`）。

## 5. 结果输出结构（stateEstimates）

滤波输出结构在 `lmb/runLmbFilter.m`、`lmbm/runLmbmFilter.m`、`multisensorLmb/runParallelUpdateLmbFilter.m`、`multisensorLmb/runIcLmbFilter.m`、`multisensorLmbm/runMultisensorLmbmFilter.m` 中一致：

- `stateEstimates.labels{t}`：2xN 标签（出生时间 + 出生位置索引）
- `stateEstimates.mu{t}` / `stateEstimates.Sigma{t}`：瞬时估计
- `stateEstimates.objects`：完整轨迹集合（含 `trajectory` 和 `timestamps`）

这些字段被 `common/plotResults.m` 和 `common/plotMultisensorResults.m` 直接使用。

## 6. 数据关联与滤波链路（代码定位）

### 6.1 单传感器 LMB（核心链路）

`lmb/runLmbFilter.m` 内部流程：

1. 预测：`lmb/lmbPredictionStep.m`
2. 生成关联矩阵：`lmb/generateLmbAssociationMatrices.m`
3. 关联算法：
   - LBP：`common/loopyBeliefPropagation.m`
   - LBPFixed：`common/fixedLoopyBeliefPropagation.m`
   - Gibbs：`lmb/lmbGibbsSampling.m`
   - Murty：`lmb/lmbMurtysAlgorithm.m`
4. 更新空间分布：`lmb/computePosteriorLmbSpatialDistributions.m`
5. 轨迹门控与 MAP 提取：`common/lmbMapCardinalityEstimate.m`

### 6.2 单传感器 LMBM

入口 `lmbm/runLmbmFilter.m`，核心流程：

1. 预测：`lmbm/lmbmPredictionStep.m`
2. 关联矩阵：`lmbm/generateLmbmAssociationMatrices.m`
3. Gibbs / Murty：`lmbm/lmbmGibbsSampling.m` 或 `common/murtysAlgorithmWrapper.m`
4. 假设更新：`lmbm/determinePosteriorHypothesisParameters.m`
5. 归一化与门控：`lmbm/lmbmNormalisationAndGating.m`
6. 状态提取：`lmbm/lmbmStateExtraction.m`

### 6.3 多传感器 LMB

- 并行更新 / AA / GA：`multisensorLmb/runParallelUpdateLmbFilter.m`
  - 传感器关联矩阵：`multisensorLmb/generateLmbSensorAssociationMatrices.m`
  - 轨迹融合：`multisensorLmb/puLmbTrackMerging.m`、`multisensorLmb/aaLmbTrackMerging.m`、`multisensorLmb/gaLmbTrackMerging.m`
- 逐传感器更新（IC）：`multisensorLmb/runIcLmbFilter.m`

### 6.4 多传感器 LMBM

入口 `multisensorLmbm/runMultisensorLmbmFilter.m`：

- 关联矩阵：`multisensorLmbm/generateMultisensorLmbmAssociationMatrices.m`
- Gibbs 采样：`multisensorLmbm/multisensorLmbmGibbsSampling.m`
- 假设更新：`multisensorLmbm/determineMultisensorPosteriorHypothesisParameters.m`

## 7. 评估与分析

### 7.1 OSPA 与跨节点共识误差指标

`common/computeSimulationOspa.m` 根据 `groundTruthRfs` 与 `stateEstimates` 计算 E-OSPA / H-OSPA，并在 `common/plotResults.m` 与 `common/plotMultisensorResults.m` 中绘图。

在“每个传感器本地融合”的分布式场景中（参见 `runMultisensorFilters_formation.m`），还会输出以下跨节点 disagreement / consensus-error 指标。这里的 consistency 不是 NEES/NIS 意义上的滤波统计一致性，也不是 label consistency：
- 单节点精度：每个传感器的 E-OSPA 与集合 RMSE（对齐公共目标后计算）
- OSPA consensus error：两两 OSPA 的平均值（同时反映位置与基数差异）
- matched localization disagreement：两两 localization RMSE 的平均值，仅在共同发现目标上计算（Hungarian 匹配）
- cardinality dispersion：各节点基数对中位数的平均绝对偏差

### 7.2 性能评测脚本

- 综合分析：`runAdvancedPerformanceAnalysis.m`（支持 name-value 参数）
- 试验脚本：`trials/` 目录（如 `trials/singleSensorAccuracyTrial.m`、`trials/multiSensorAccuracyTrial.m`）

示例：

```matlab
report = runAdvancedPerformanceAnalysis('ClutterRates', [5 10], 'NumberOfTrials', 10);
```

## 8. 常见注意事项

- LMBM 不支持 LBP（在 `common/generateModel.m` / `common/generateMultisensorModel.m` 的注释中已说明）。
- 多传感器 LMBM 在 `multisensorLmbm/runMultisensorLmbmFilter.m` 中有“极慢且占内存”的警告，建议先用小规模验证。
- measurements 的格式严格依赖 cell 结构，若你是从外部数据导入，请确保与 `common/generateGroundTruth.m` / `common/generateMultisensorGroundTruth.m` 产出一致。

## 9. 推荐阅读顺序（代码定位版）

1. `runFilters.m` 或 `runMultisensorFilters.m`：整体流程
2. `common/generateModel.m` / `common/generateMultisensorModel.m`：模型参数字段
3. `lmb/runLmbFilter.m` 或 `multisensorLmb/runParallelUpdateLmbFilter.m`：核心滤波链路
4. `common/plotResults.m` / `common/plotMultisensorResults.m`：输出与评价
5. `runAdvancedPerformanceAnalysis.m` 与 `trials/`：性能比较与分析
