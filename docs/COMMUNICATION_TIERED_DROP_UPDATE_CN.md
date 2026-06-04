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

围绕这套通信配置，当前脚本更适合分成“主线脚本”和“次线脚本”两组来看。

主线脚本：

- [RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m](../RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m)
  `fixed -> +covariance -> +link quality -> +existence confidence -> +structure-aware decoupled KLA`
- [RUN/GA/runMultisensorFilters_formation_4plus4_IdealCommCompare.m](../RUN/GA/runMultisensorFilters_formation_4plus4_IdealCommCompare.m)
  ideal communication 下 `ordinary GA -> structure-aware decoupled KLA`

次线或附录脚本：

- [RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkNISCompare.m](../RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkNISCompare.m)
  `w/o NIS -> robust NIS -> NIS`
- [RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkFreshnessCompare.m](../RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkFreshnessCompare.m)
  `robust NIS baseline -> robust NIS + freshness`
- [RUN/GA/runMultisensorFilters_formation_4plus4_HistoryCompare.m](../RUN/GA/runMultisensorFilters_formation_4plus4_HistoryCompare.m)
  `w/o history -> history`

这些脚本共享同一组 tiered communication 口径，但当前论文或主文档更建议优先围绕第一组脚本组织叙事。

## 6. 当前验证结论

以下结果都来自 `5 trial`，并使用同一套分档通信配置。

### 6.1 当前主线结论：四项组合是最稳的叙事

主线报告：

- [RUN/GA/GA_TIERED_LINK_ABLATION_20260322_001613.md](../RUN/GA/GA_TIERED_LINK_ABLATION_20260322_001613.md)
- [RUN/GA/GA_TIERED_LINK_ABLATION_20260326_182435.md](../RUN/GA/GA_TIERED_LINK_ABLATION_20260326_182435.md)

当前主线消融路径：

```text
fixed weights -> +covariance -> +link quality -> +existence confidence -> +structure-aware decoupled KLA
```

当前可直接引用的一组结果是：

```text
fixed weights:                  OSPA 2.624065, RMSE 2.702602, Card 0.878750
+covariance:                    OSPA 2.211513, RMSE 2.410976, Card 0.589500
+link quality:                  OSPA 1.877771, RMSE 1.800945, Card 0.245250
+existence confidence:          OSPA 1.874840, RMSE 1.779820, Card 0.244500
+structure-aware decoupled KLA: OSPA 1.862244, RMSE 1.749608, Card 0.244250
```

这条路径现在已经比旧的 “adaptive robust NIS” 叙事更适合做主线，因为它的每一步边际作用都更清楚。

### 6.2 `covariance + link quality` 是主线骨架

结论：

- `covariance` 先把 fixed weights 明显拉开，说明 posterior quality 本身必须进权重
- `link quality` 在 tiered-drop 条件下继续带来最大一档附加收益，说明通信异构性必须被显式建模
- 这两项一起构成当前 adaptive weighting 的基础骨架

### 6.3 `existence confidence` 是最有效的第三因子

报告：

- [RUN/GA/GA_TIERED_LINK_ABLATION_20260322_001613.md](../RUN/GA/GA_TIERED_LINK_ABLATION_20260322_001613.md)

这次测试的是在 `协方差 + 链路质量` 基础上，再加入一个新的“存在性/基数置信度”因子。它不是看链路是否丢包，也不是看协方差大小，而是看各 Bernoulli existence probability 是否足够尖锐。

推荐参数是：

```matlab
model.adaptiveFusion.useExistenceConfidence = true;
model.adaptiveFusion.existenceConfidenceMinScore = 0.85;
model.adaptiveFusion.existenceConfidencePower = 2.0;
model.adaptiveFusion.useNIS = false;
```

`5 trial` 结果：

- `+link quality`: `OSPA 1.877771`, `RMSE 1.800945`, `Cardinality 0.245250`
- `+existence confidence`: `OSPA 1.874840`, `RMSE 1.779820`, `Cardinality 0.244500`

结论：

- 这是目前第一个相对 `协方差 + 链路质量` baseline 稳定实现三项指标同时改善的新因子
- 它比 `freshness` 更有效，也比 `robust NIS` 更适合放在主线第三个位置
- 从论文表述上，它补充的是“存在性判决可信度”，与“状态精度”和“通信可靠性”形成互补

### 6.4 当前最优：weak structure-aware decoupled KLA

报告：

- [RUN/GA/GA_TIERED_LINK_ABLATION_20260326_182435.md](../RUN/GA/GA_TIERED_LINK_ABLATION_20260326_182435.md)

这轮是在 `协方差 + 链路质量 + existence confidence` 的基础上，进一步做一个很弱的 structure-aware decoupled KLA：

- spatial 分支保留主要收益
- existence 分支只做很轻的结构调制，避免破坏 cardinality
- 结构先验同时参考局部子图重叠和固定分档丢包率
- posterior-consistency 结构模式保留为实验开关，但当前 best 默认仍使用静态结构先验模式

推荐参数是：

```matlab
model.adaptiveFusion.useCovariance = true;
model.adaptiveFusion.useLinkQuality = true;
model.adaptiveFusion.useExistenceConfidence = true;
model.adaptiveFusion.existenceConfidenceMinScore = 0.85;
model.adaptiveFusion.existenceConfidencePower = 2.0;
model.adaptiveFusion.useDecoupledKla = true;
model.adaptiveFusion.useStructureAwareKla = true;
model.adaptiveFusion.usePosteriorStructureConsistency = false;
model.adaptiveFusion.spatialDecouplingStrength = 0.5;
model.adaptiveFusion.existenceDecouplingStrength = 0.15;
model.adaptiveFusion.spatialStructureStrength = 0.45;
model.adaptiveFusion.existenceStructureStrength = 0.08;
model.adaptiveFusion.structureReliabilityPower = 0.30;
model.adaptiveFusion.useNIS = false;
```

`5 trial` 结果：

- `+link quality`: `OSPA 1.877771`, `RMSE 1.800945`, `Cardinality 0.245250`
- `+structure-aware decoupled KLA`: `OSPA 1.862244`, `RMSE 1.749608`, `Cardinality 0.244250`

结论：

- 这是当前 tiered 通信配置下的最新 best
- 相比 `+existence confidence` baseline，`OSPA`、`RMSE` 和 `Cardinality` 都继续下降
- 当前有效配置的关键不是“强结构先验”，而是“在三因子 baseline 上叠加很弱的 structure-aware decoupling”
- 这也意味着 structure-aware 更适合作为 refinement 来写，而不是单独写成拓扑权重主方法

### 6.5 ideal communication 下的支持性证据

报告：

- [RUN/GA/GA_IDEAL_COMM_COMPARE_20260326_184508.md](../RUN/GA/GA_IDEAL_COMM_COMPARE_20260326_184508.md)

结果：

```text
ordinary GA -> structure-aware decoupled KLA
OSPA consensus error: 1.706 -> 1.494
Matched localization disagreement: 1.526 -> 1.290
Cardinality dispersion: 0.161 -> 0.139
Local E-OSPA:   1.950 -> 1.877
Local RMSE:     1.442 -> 1.369
```

结论：

- 结构解耦层的收益不只是“补偿链路丢包”
- 即使在 ideal communication 下，它对普通 GA 也仍有正向作用
- 这条结果适合作为 supporting evidence，而不是替代 tiered main scenario

### 6.6 次线与负结果：`robust NIS`、`freshness`、`history`

`robust NIS` 报告：

- [RUN/GA/GA_TIERED_LINK_NIS_COMPARE_20260321_193628.md](../RUN/GA/GA_TIERED_LINK_NIS_COMPARE_20260321_193628.md)

结果：

```text
w/o NIS -> robust NIS -> NIS
OSPA: 1.909267 -> 1.908967 -> 2.007700
RMSE: 2.934317 -> 2.980071 -> 3.173222
Card: 0.267000 -> 0.262250 -> 0.300500
```

结论：

- `robust NIS` 比 plain `NIS` 稳定
- 但它与 `w/o NIS` 基本打平，不足以构成当前主线 headline

`freshness` 报告：

- [RUN/GA/Del_GA_TIERED_LINK_FRESHNESS_COMPARE_20260321_193131.md](../RUN/GA/Del_GA_TIERED_LINK_FRESHNESS_COMPARE_20260321_193131.md)

结果：

```text
OSPA: 1.908967 -> 1.909680
RMSE: 2.980071 -> 2.979829
Card: 0.262250 -> 0.262500
```

结论：

- `freshness` 在当前口径下几乎不起作用
- 只适合保留成负结果或附录说明

`history` 报告：

- [RUN/GA/GA_HISTORY_COMPARE_20260309_113545.md](../RUN/GA/GA_HISTORY_COMPARE_20260309_113545.md)

结果：

```text
w/o history -> history
OSPA: 1.811 -> 1.814
RMSE: 3.173 -> 3.158
Card: 0.214 -> 0.215
```

结论：

- `history` 只带来很弱而且带耦合的变化
- 更适合作为“尝试过，但不进正文主线”的补充材料

## 7. 推荐口径

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

如果后续要在这套通信配置下给出当前最佳动态权重组合，建议优先采用：

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

对应主报告为：

- [RUN/GA/GA_TIERED_LINK_ABLATION_20260326_182435.md](../RUN/GA/GA_TIERED_LINK_ABLATION_20260326_182435.md)

这套组合当前对应的是：

- `协方差`：状态精度
- `链路质量`：通信可靠性
- `存在性置信度`：目标存在性/基数判决可信度
- `弱结构先验解耦`：对 spatial 分支做主要 refinement，并只对 existence 分支做轻微调制

如果需要写论文正文，建议把 `robust NIS`、`history`、`freshness` 统一挪到次线或附录，而不要再和这四项主线并列叙述。
