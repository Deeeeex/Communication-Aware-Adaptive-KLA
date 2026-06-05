# 通信配置改动记录：分档丢包模型

本文档记录本轮通信层配置上的改动，重点说明为什么从“统一固定丢包率”切换到“分档异构丢包率”，以及这项改动在当前 formation `4+4` 实验中的使用方式和初步结果。

## 1. 改动背景

此前 Level 2 的默认链路模型是：

```matlab
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
```

这意味着：

- 所有通过带宽筛选的传感器包，都使用同一个标量 `pDrop`
- 各节点在统计意义上是同分布掉包
- `linkQuality` 虽然会随时间变化，但节点间长期差异较弱

这类配置适合做“整体通信变差”的压力测试，但不利于体现“不同节点链路质量不同”这一类真实网络异构性。

从自适应权重的角度看，如果所有节点的掉包机制本来就几乎一样，那么 `linkQuality` 这个因子的可区分度会被压平。

## 2. 这次改了什么

本轮改动在 [common/applyCommunicationModel.m](../common/applyCommunicationModel.m) 中给 `fixed` 链路模式增加了两类新入口：

- `pDropBySensor`
  直接给出每个传感器的固定丢包率向量
- `pDropLevels + pDropLevelCounts`
  用离散档位和档位个数自动生成 `pDropBySensor`

解析优先级是：

1. `pDropBySensor`
2. `pDropLevels + pDropLevelCounts`
3. 标量 `pDrop`

也就是说，这次并没有推翻旧的 `pDrop` 写法，只是在其上增加了更强的异构配置能力。

## 3. 分档丢包模型的设计

### 3.1 设计目标

目标不是简单地“把每个传感器随机设成不同的连续丢包率”，而是构造一个更容易解释、更接近工程直觉的链路分层：

- 好链路
- 较好链路
- 一般链路
- 差链路

因此采用分档设计，而不是连续随机设计。

### 3.2 当前默认档位

当前 formation `4+4` 实验里使用：

```matlab
commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
commConfig.pDropLevelCounts = [1, 4, 1, 2];
```

对应含义：

- 1 个节点：`pDrop = 0`
- 4 个节点：`pDrop = 0.1`
- 1 个节点：`pDrop = 0.2`
- 2 个节点：`pDrop = 0.5`

对于 8 传感器场景，这组配置的平均值是：

$$
\frac{1 \times 0 + 4 \times 0.1 + 1 \times 0.2 + 2 \times 0.5}{8} = 0.2
$$

因此它与历史上常用的 `pDrop = 0.2` 在总体通信强度上是对齐的。

### 3.3 Trial 内与 Trial 间的行为

当前实现中：

- 每个 trial 开始时，会按 `pDropLevelCounts` 先生成一组长度为传感器数的 `pDropBySensor`
- 再对这组档位做一次随机打散，分配给各传感器
- 一旦分配完成，该 trial 内保持不变

这样做的原因是：

- trial 内固定，便于解释“某个节点长期链路较差”的效应
- trial 间打散，避免总是同一编号节点吃亏

## 4. 与旧配置相比的变化点

### 4.1 旧配置：统一固定丢包率

```matlab
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
```

特点：

- 节点间同分布
- 易于实现
- 适合验证“整体通信压力”
- 不容易拉开自适应链路质量因子的作用

### 4.2 中间尝试：连续异构丢包率

在本轮探索中还测试过“每个 trial 随机生成一组连续 `pDropBySensor`，但总体平均仍约束在 `0.2` 附近”的方案。

特点：

- 异构性比统一 `pDrop` 强
- 能提升 `linkQuality` 的辨识度
- 但可解释性一般，不如分档清楚

### 4.3 当前推荐：分档异构丢包率

```matlab
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
commConfig.pDropLevelCounts = [1, 4, 1, 2];
```

特点：

- 总体平均仍然和旧配置对齐
- 节点间存在清晰的离散层级
- 更接近“网络里有好链路也有坏链路”的直觉
- 更适合作为论文或实验文档中的通信设置说明

## 5. 当前实验脚本

围绕这套通信配置，clean 版本只保留论文正文采用的 GA 主线脚本：

- [RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m](../RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m)
  `Fixed Metropolis -> PD-weighted GA -> FID-FIA-weighted GA -> Balanced mode -> Cardinality-critical mode`
- [RUN/GA/runMultisensorFilters_formation_4plus4_IdealCommCompare.m](../RUN/GA/runMultisensorFilters_formation_4plus4_IdealCommCompare.m)
  ideal communication 下复用同一组 5 个 paper-facing arms
- [RUN/GA/runMultisensorFilters_formation_4plus4_CommLevelThreeMethodCompare.m](../RUN/GA/runMultisensorFilters_formation_4plus4_CommLevelThreeMethodCompare.m)
  通信等级 sensitivity：`Fixed Metropolis -> Balanced mode -> Cardinality-critical mode`

NIS、history、freshness、association ambiguity、posterior-structure consistency 等历史尝试不再作为 clean 版本的可运行 GA 入口维护。

## 6. Clean 版实验口径

本仓库不保留中间报告和论文源码，只保留可复现实验入口。GA 主实验默认使用
论文正文表格的组织形式。

### 6.1 主表对比

默认入口：

```matlab
[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, true, struct(), writeReport, ...
    'fidFiaExistenceRefinement');
```

默认 arm 顺序：

```text
Fixed Metropolis -> PD-weighted GA -> FID-FIA-weighted GA ->
Balanced mode -> Cardinality-critical mode
```

### 6.2 因子消融

如果需要论文正文中的 backbone/final-mode 因子消融，使用：

```matlab
[reportPath, summary] = runMultisensorFilters_formation_4plus4_TieredLinkAblation( ...
    numberOfTrials, baseSeed, true, struct(), writeReport, ...
    'factorAblation');
```

对应 arm 顺序：

```text
Fixed Metropolis -> Covariance-only adaptive -> Covariance-link adaptive ->
Balanced mode -> Cardinality-critical mode
```

### 6.3 Ideal Communication

Ideal communication 支撑实验不再维护单独的一套旧 arm，而是复用主表的
5 个 paper-facing arms：

```matlab
[reportPath, summary] = runMultisensorFilters_formation_4plus4_IdealCommCompare( ...
    numberOfTrials, baseSeed, true, writeReport);
```

### 6.4 通信等级敏感性

通信等级 sensitivity 只比较正文采用的三个 representative arms：

```text
Fixed Metropolis -> Balanced mode -> Cardinality-critical mode
```

对应入口：

```matlab
summaries = runMultisensorFilters_formation_4plus4_CommLevelThreeMethodCompare( ...
    numberOfTrials, baseSeed, true, writeReports);
```

## 7. 推荐通信配置

如果后续文档、实验或论文需要统一通信设置，建议优先使用分档异构口径：

```matlab
commConfig.level = 2;
commConfig.globalMaxMeasurementsPerStep = 80;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'random';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
commConfig.pDropLevelCounts = [1, 4, 1, 2];
commConfig.maxOutageNodes = 1;
```

推荐理由：

- 与历史 `pDrop=0.2` 的总体强度可对齐
- 对 `linkQuality` 更敏感
- 更容易解释实验现象
- 更方便后续扩展到“档位个数变化”或“极差节点个数变化”的对比实验

如果后续要在这套通信配置下手动构造 Balanced mode，建议优先采用：

```matlab
model.adaptiveFusion.useCovariance = true;
model.adaptiveFusion.useLinkQuality = true;
model.adaptiveFusion.useExistenceConfidence = true;
model.adaptiveFusion.existenceConfidenceMinScore = 0.85;
model.adaptiveFusion.existenceConfidencePower = 2.0;
model.adaptiveFusion.useDecoupledKla = true;
model.adaptiveFusion.useStructureAwareKla = true;
model.adaptiveFusion.spatialDecouplingStrength = 0.5;
model.adaptiveFusion.existenceDecouplingStrength = 0.15;
model.adaptiveFusion.spatialStructureStrength = 0.45;
model.adaptiveFusion.existenceStructureStrength = 0.08;
model.adaptiveFusion.structureReliabilityPower = 0.30;
model.adaptiveFusion.useNIS = false;
model.adaptiveFusion.useHistory = false;
```

这套组合当前对应的是：

- `协方差`：状态精度
- `链路质量`：通信可靠性
- `存在性置信度`：目标存在性/基数判决可信度
- `弱结构先验解耦`：对 spatial 分支做主要 refinement，并只对 existence 分支做轻微调制

Cardinality-critical mode 在 Balanced mode 基础上只给 existence branch
增加 FID-FIA refinement：

```matlab
model.adaptiveFusion.useFidFiaExistence = true;
model.adaptiveFusion.fidFiaExistenceStrength = 4.0;
model.adaptiveFusion.fidFiaExistenceMinScore = 0.0;
model.adaptiveFusion.existenceEmaAlpha = 0.0;
model.adaptiveFusion.existenceMinWeight = 0.0;
```
