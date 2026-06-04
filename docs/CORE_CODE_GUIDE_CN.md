# 核心代码导读

本文档整理仓库的核心代码边界和推荐阅读顺序。这里的“核心代码”指会被仿真、滤波、融合和评估流程直接复用的库代码与少量入口脚本，不包括论文模板、历史实验报告、生成图表、PPT 输出和一次性批处理产物。

## 推荐阅读顺序

1. 先看入口脚本：`setPath.m`、`runFilters.m`、`runMultisensorFilters.m`、`runMultisensorFilters_formation.m`。
2. 再看模型和数据生成：`common/generateModel.m`、`common/generateMultisensorModel.m`、`common/generateGroundTruth.m`、`common/generateMultisensorGroundTruth.m`。
3. 接着看单传感器 LMB 主链路：`lmb/runLmbFilter.m`、`lmb/lmbPredictionStep.m`、`lmb/generateLmbAssociationMatrices.m`、`lmb/computePosteriorLmbSpatialDistributions.m`。
4. 然后看多传感器融合：`multisensorLmb/runParallelUpdateLmbFilter.m`、`multisensorLmb/generateLmbSensorAssociationMatrices.m`、`multisensorLmb/gaLmbTrackMerging.m`、`multisensorLmb/aaLmbTrackMerging.m`、`multisensorLmb/puLmbTrackMerging.m`。
5. 最后看本文工作的扩展重点：`common/evaluateSensorQuality.m`、`common/applyCommunicationModel.m`、`common/applyMultiRateSensorSchedule.m`、`multisensorLmb/computeAdaptiveFusionWeights.m`、`multisensorLmb/runDistributedLmbFilter.m`。

## 主链路

典型多传感器实验的数据流是：

```text
入口脚本
  -> 生成 model
  -> 生成 ground truth / measurements / sensorTrajectories
  -> 可选通信约束和多速率调度
  -> LMB prediction
  -> 每个传感器构造 association matrices
  -> LBP/Gibbs/Murty 数据关联
  -> 每个传感器形成 measurement-updated posterior
  -> PU/GA/AA 或自适应加权融合
  -> MAP cardinality extraction
  -> OSPA/图表/实验报告
```

## 入口与实验脚本

| 文件 | 导读 |
| --- | --- |
| `setPath.m` | MATLAB 会话初始化脚本，只把源码目录加入 path。 |
| `runFilters.m` | 最小单传感器示例，用于快速检查 LMB/LMBM 基线是否能跑通。 |
| `runMultisensorFilters.m` | 紧凑的集中式多传感器示例，覆盖模型、通信约束和 IC/PU/GA/AA/LMBM 分支。 |
| `runMultisensorFilters_formation.m` | 移动传感器、FOV、通信约束、分布式邻居融合和自适应权重的综合演示。 |
| `runMultisensorFilters_mobile.m` | 更小的移动传感器示例，适合先调 sensor-relative measurement 逻辑。 |
| `RUN/GA/*.m`、`RUN/AA/*.m`、`RUN/IDEAL/*.m` | 消融和对照实验入口。它们重要，但多数是参数编排脚本，不是可复用库核心。 |

## 模型、场景和通信层

| 文件 | 导读 |
| --- | --- |
| `common/generateModel.m` | 单传感器模型工厂，集中定义运动、观测、birth、clutter、门限和关联参数。 |
| `common/generateGroundTruth.m` | 单传感器轨迹、测量、clutter 和 RFS truth 生成器。 |
| `common/generateMultisensorModel.m` | 多传感器模型工厂，增加每个传感器的噪声、检测率、移动/FOV、formation 和 PU/GA/AA 权重。 |
| `common/generateMultisensorModelEnhanced.m` | 兼容旧 enhanced-motion 入口的包装层；当前主实现主要在 `generateMultisensorModel.m`。 |
| `common/generateMultisensorGroundTruth.m` | 多传感器轨迹、传感器轨迹、state-dependent quality、clutter 和 RFS truth 生成器。 |
| `common/generateMultisensorGroundTruthEnhanced.m` | 兼容旧脚本的包装层，直接委托到 `generateMultisensorGroundTruth.m`。 |
| `common/evaluateSensorQuality.m` | 几何/FOV 导致的传感质量变化中心：统一计算每个传感器的 `p_D`、测量协方差和诊断信息。 |
| `common/applyCommunicationModel.m` | 通信约束层：带宽、链路丢包、节点 outage、可用性 mask 和 link quality 统计。 |
| `common/applyMultiRateSensorSchedule.m` | 多速率传感器调度层：将未采样时刻从测量流里剔除，并记录 sample age/freshness。 |

## 单传感器 LMB

| 文件 | 导读 |
| --- | --- |
| `lmb/runLmbFilter.m` | 单传感器 LMB 主循环：prediction、association、posterior update、prune、MAP extraction。 |
| `lmb/lmbPredictionStep.m` | LMB prediction，共用在单传感器和多传感器 LMB 流程中。 |
| `lmb/generateLmbAssociationMatrices.m` | 单传感器 measurement update 前端，生成 LBP/Gibbs/Murty 共用的关联矩阵和 Kalman 更新组件。 |
| `lmb/computePosteriorLmbSpatialDistributions.m` | 用关联边缘概率重组 posterior Gaussian mixture，并做 mixture pruning/capping。 |
| `common/loopyBeliefPropagation.m` | 默认快速关联求解器，返回 posterior existence 和 marginal association weights。 |
| `common/fixedLoopyBeliefPropagation.m` | 固定迭代数 LBP，主要用于复杂度/运行时对照。 |
| `lmb/lmbGibbsSampling.m` | Gibbs 采样关联后端，适合随机近似或不确定性对照。 |
| `lmb/lmbMurtysAlgorithm.m` | K-best assignment 关联后端，适合小规模或基准比较。 |
| `common/lmbMapCardinalityEstimate.m` | LMB 输出阶段的 MAP cardinality 和组件选择。 |

## 多传感器 LMB

| 文件 | 导读 |
| --- | --- |
| `multisensorLmb/runParallelUpdateLmbFilter.m` | 集中式多传感器 LMB 主循环；每个传感器本地更新后用 PU/GA/AA 融合。 |
| `multisensorLmb/runIcLmbFilter.m` | Iterated-corrector baseline，同一时刻内按传感器顺序逐次更新 posterior。 |
| `multisensorLmb/runDistributedLmbFilter.m` | 分布式封装：为每个传感器构造邻域 sub-model，再调用并行更新主循环。 |
| `multisensorLmb/generateLmbSensorAssociationMatrices.m` | 每个传感器的关联矩阵构造，支持移动几何、state-dependent `p_D/Q`、NIS 和 ambiguity 统计。 |
| `multisensorLmb/puLmbTrackMerging.m` | PU 融合，按条件独立传感器假设合并 local posteriors。 |
| `multisensorLmb/gaLmbTrackMerging.m` | GA 融合，按空间权重和存在概率权重做几何平均。 |
| `multisensorLmb/aaLmbTrackMerging.m` | AA 融合，拼接并截断 Gaussian mixture，保留更多多峰结构。 |
| `multisensorLmb/computeAdaptiveFusionWeights.m` | 自适应 GA/AA 权重核心：factorized 主线、PD-weighted GA、FI-weighted GA、FID-FIA、decoupled KLA、structure-aware 和 Cardinality-critical refinement 都在这里。 |

## LMBM 参考实现

| 文件 | 导读 |
| --- | --- |
| `lmbm/runLmbmFilter.m` | 单传感器 LMBM 主循环，保留多个 global hypotheses，计算更重。 |
| `lmbm/generateLmbmAssociationMatrices.m` | 单传感器 LMBM 的 hypothesis-level 关联矩阵构造。 |
| `lmbm/determinePosteriorHypothesisParameters.m` | 将关联事件转换为 posterior global hypotheses。 |
| `lmbm/lmbmNormalisationAndGating.m` | LMBM hypothesis pruning 和低存在概率组件剔除。 |
| `lmbm/lmbmPredictionStep.m` | LMBM hypothesis prediction 和 birth 追加。 |
| `lmbm/lmbmStateExtraction.m` | 从最高权重 hypothesis 中抽取状态估计。 |
| `multisensorLmbm/runMultisensorLmbmFilter.m` | 多传感器 LMBM 参考实现，组合空间很大，主要用于小规模对照。 |
| `multisensorLmbm/generateMultisensorLmbmAssociationMatrices.m` | 多传感器 LMBM 的高维关联似然构造。 |
| `multisensorLmbm/determineMultisensorPosteriorHypothesisParameters.m` | 多传感器 LMBM posterior hypothesis 重建。 |

## 评估、绘图和非核心目录

| 路径 | 角色 |
| --- | --- |
| `common/computeSimulationOspa.m`、`common/ospa.m` | 核心评估指标实现。 |
| `common/plotResults.m`、`common/plotMultisensorResults.m`、`visualizeFilterPerformance.m` | 可视化辅助，不影响滤波核心逻辑。 |
| `trials/*.m` | 运行时、精度、clutter 和检测率试验脚本。 |
| `marginalEvalulations/*.m` | 边缘分布/关联矩阵的小实验辅助。注意目录名沿用了已有拼写。 |

## 修改代码时的边界

- 改滤波行为时，优先从 `lmb/`、`multisensorLmb/` 和 `common/` 的核心函数入手。
- 改实验设定时，优先改入口脚本或 `RUN/*/run*.m`，不要把一次性参数硬编码进核心库函数。
- `computeAdaptiveFusionWeights.m` 是当前最复杂的单文件，修改前先确认目标是 direct baseline、factorized mode、decoupled KLA，还是 debug/report 字段。
