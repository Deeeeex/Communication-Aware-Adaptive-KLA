# Tiered Link Ablation 调用链导读

本文档选取 `RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m`，按一次完整实验从入口到动态权重、融合、指标和报告的调用链条解释代码逻辑。

选择这个脚本的原因：

- 它是当前 GA-LMB 动态权重主实验/消融入口；
- 它覆盖 Fixed Metropolis、FID-FIA baseline、Balanced mode 和 Cardinality-critical mode；
- 它能串起 `RUN/GA` 实验脚本、分布式 LMB、动态权重核心和 GA merging 端。

## 1. 总调用链

```text
runMultisensorFilters_formation_4plus4_TieredLinkAblation
  -> generateMultisensorModel
  -> generateMultisensorGroundTruth
  -> applyOptionalMultiRateSchedule
  -> applyCommunicationModel
  -> buildArms
  -> for each arm:
       -> runDistributedLmbFilter
          -> build local model for each sensor neighborhood
          -> runParallelUpdateLmbFilter
             -> generateLmbSensorAssociationMatrices
             -> computePosteriorLmbSpatialDistributions
             -> computeAdaptiveFusionWeights
             -> gaLmbTrackMerging
       -> computeSimulationOspa
       -> computeSetRmseOverTime
       -> computeConsensusMetrics
  -> writeAblationReport
```

这条链条里，实验脚本只负责组织实验；动态权重的真实计算在 `computeAdaptiveFusionWeights.m`；权重真正落到 posterior fusion 公式里是在 `gaLmbTrackMerging.m`。

## 2. 入口参数

函数签名：

```matlab
[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, useFixedSeed, commConfigOverrides, ...
    writeReport, finalArmMode, adaptiveFusionOverrides, armSelection)
```

关键参数含义：

| 参数 | 作用 |
| --- | --- |
| `numberOfTrials` | Monte Carlo trial 数。每个 trial 生成一份共享场景，所有 arm 在同一场景上配对比较。 |
| `baseSeed` / `useFixedSeed` | 控制随机种子，保证不同 arm 在同一 trial 内可比。 |
| `commConfigOverrides` | 覆盖通信配置，比如丢包等级、多速率采样等。 |
| `writeReport` | 是否写出 markdown 报告。调试 smoke test 通常设为 `false`。 |
| `finalArmMode` | 决定 buildArms 构造哪组对比。当前主线用 `fidFiaExistenceRefinement`。 |
| `adaptiveFusionOverrides` | 覆盖所有 arm 的基础动态权重参数。 |
| `armSelection` | 临时只跑部分 arm 的调试入口，不改变默认实验定义。 |

## 3. 场景生成

脚本先准备四类共享配置：

| 配置 | 代码变量 | 作用 |
| --- | --- | --- |
| 动态权重基础配置 | `baseAdaptiveFusionConfig` | 所有 arm 的默认 adaptiveFusion 参数。 |
| 通信配置 | `commConfig` | tiered packet drop、bandwidth、outage 等通信限制。 |
| 传感器运动 | `sensorMotionConfig` | 8 个传感器的 CV 运动配置。 |
| 目标编队 | `targetFormationConfig` | staggered births、目标出生状态和生命周期。 |

进入 trial 后，代码调用：

```matlab
model = generateMultisensorModel(...);
[~, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model);
```

此时还没有动态权重。这里得到的是完整 truth、原始 measurements 和传感器轨迹。

## 4. 通信注入

通信层调用顺序：

```matlab
[measurementsForComm, samplingStats] = applyOptionalMultiRateSchedule(measurements, commConfig);
[measurementsDelivered, commStats] = applyCommunicationModel(measurementsForComm, model, commConfig);
commStats = attachSamplingStats(commStats, samplingStats);
```

`measurementsDelivered` 是滤波器真正看到的测量。`commStats` 是动态权重和报告共用的通信诊断来源，尤其包括：

- `pDropBySensor`
- `droppedByBandwidth`
- `droppedByLink`
- `droppedByOutage`
- `fusionMask` 或 `activeMask`

当前动态权重主线里，`linkQuality` 和 `availabilityMask` 就依赖这些通信统计。

## 5. Arm 构造

`buildArms(baseAdaptiveFusionConfig, finalArmMode)` 是实验设计核心。

当前 paper 主线：

```matlab
finalArmMode = 'fidFiaExistenceRefinement'
```

对应 arm 顺序：

| Arm | 实验含义 |
| --- | --- |
| `fixed weights` | 关闭 adaptiveFusion，使用固定 Metropolis 拓扑权重。 |
| `Cao-Zhao FID-FIA baseline` | 用 scalar FID-FIA score 直接生成 GA 权重。 |
| `+structure-aware decoupled KLA` | Balanced mode：三因子 backbone + spatial/existence decoupling + structure prior。 |
| `+FID-FIA existence refinement` | Cardinality-critical mode：在 Balanced 基础上，只给 existence branch 加 FID-FIA。 |

Cardinality-critical 的关键参数：

```matlab
cfg.useFidFiaExistence = true;
cfg.fidFiaExistenceStrength = 4.0;
cfg.fidFiaExistenceMinScore = 0.0;
cfg.existenceEmaAlpha = 0.0;
cfg.existenceMinWeight = 0.0;
```

这表示 FID-FIA 不替换 spatial 权重，只强力调制 existence/cardinality 分支。

## 6. 进入分布式滤波

每个 arm 调用：

```matlab
[stateEstimatesBySensor, localModels] = runDistributedLmbFilter( ...
    armModel, measurementsDelivered, sensorTrajectories, neighborMap, commStats);
```

这一步把全局 8-sensor 问题拆成每个 sensor 的邻域局部问题：

- `buildNeighborMap4Plus4` 给出 4+4 分组加跨组配对邻居；
- `runDistributedLmbFilter` 为每个 sensor 构造 local model；
- local model 会带上邻域对应的 `gaSensorWeights`、`gaSpatialWeights`、`gaExistenceWeights` 和 structure priors；
- 每个 local model 再进入 `runParallelUpdateLmbFilter`。

## 7. 动态权重真正发生的位置

`runParallelUpdateLmbFilter` 每个时刻先做 local posterior update：

```matlab
associationMatrices = generateLmbSensorAssociationMatrices(...);
measurementUpdatedDistributions{s} = computePosteriorLmbSpatialDistributions(...);
```

随后调用：

```matlab
[gaWeights, aaWeights, debug] = computeAdaptiveFusionWeights(...);
```

当前主线权重逻辑：

```text
rawScore = availabilityMask * covScore * linkQuality * existenceConfidenceScore
```

如果启用 decoupled KLA，则输出两套分支权重：

- `debug.gaSpatialWeights`
- `debug.gaExistenceWeights`

如果启用 Cardinality-critical，则只改 existence branch：

```text
existenceScore = existenceScore * fidFiaExistenceScore^fidFiaExistenceStrength
```

## 8. 权重如何落到融合公式

动态权重写回 local model 后，GA 融合调用：

```matlab
gaLmbTrackMerging(measurementUpdatedDistributions, model)
```

权重消费方式：

| 分支 | 权重字段 | 融合对象 |
| --- | --- | --- |
| spatial branch | `gaSpatialWeights` | Gaussian spatial density / canonical parameters |
| existence branch | `gaExistenceWeights` | Bernoulli existence probability |
| target-wise baseline | `gaTargetWiseWeights` | PD/FI direct baseline 的 target-specific weights |

这就是为什么 Cardinality-critical 能改善 cardinality dispersion：它允许 existence branch 使用更强的 FID-FIA 调制，而 spatial branch 仍维持 Balanced mode 的定位权重。

## 9. 指标汇总

每个 arm 滤波结束后，脚本计算两类指标。

Local tracking metrics：

```matlab
computeSimulationOspa(...)
computeSetRmseOverTime(...)
```

它们比较每个 sensor 输出和 ground truth，防止方法只让节点“彼此一致”但一起偏离真值。

Network disagreement metrics：

```matlab
computeConsensusMetrics(stateEstimatesBySensor, armModel)
```

输出：

| 变量 | paper 名称 | 含义 |
| --- | --- | --- |
| `consOspa` | OSPA consensus error | sensor 输出集合之间的 OSPA disagreement。 |
| `consPos` | matched localization disagreement | sensor 输出之间 Hungarian-matched 位置 RMSE。 |
| `consCard` | cardinality dispersion | sensor 目标数相对全网 median count 的平均偏差。 |

## 10. 报告与 summary

脚本最后输出两个结果通道：

| 输出 | 用途 |
| --- | --- |
| `summary` | 给批处理脚本、后续绘图、表格合成直接读取。 |
| markdown report | 给人读，包含配置、trial 级结果、均值、置信区间、paired improvement 和 runtime。 |

报告写出函数：

```matlab
writeAblationReport(...)
```

如果 `writeReport=false`，脚本不生成 markdown，但仍返回完整 `summary`，适合 smoke test。

## 11. 最短调试命令

只跑 1 trial、不写报告、使用当前主线 arm：

```matlab
addpath('RUN/GA');
[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    1, 1, true, struct(), false, 'fidFiaExistenceRefinement');
```

如果只想验证最终 Cardinality-critical arm：

```matlab
[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    1, 1, true, struct(), false, 'fidFiaExistenceRefinement', struct(), 'final');
```

`summary.consensus` 是最先看的 smoke-test 输出。
