# 动态权重核心代码导读

本文档只整理动态融合权重相关的核心代码。按你的要求，这里暂时不展开 `common/` 里的模型生成、通信模型、OSPA、传感器质量等基础模块；这些模块只被视为动态权重的输入来源。

动态权重主链路是：

```text
RUN/GA 实验脚本配置 model.adaptiveFusion
  -> multisensorLmb/runDistributedLmbFilter.m
     为每个节点构造邻域 sub-model 和初始拓扑权重
  -> multisensorLmb/runParallelUpdateLmbFilter.m
     每步收集本地 measurement-updated LMB posterior
  -> multisensorLmb/computeAdaptiveFusionWeights.m
     计算 GA/AA 的空间权重、存在权重、target-wise 权重和 debug 因子
  -> multisensorLmb/gaLmbTrackMerging.m 或 aaLmbTrackMerging.m
     真正把权重用于 LMB posterior fusion
```

## 最核心文件

| 文件 | 角色 |
| --- | --- |
| `multisensorLmb/computeAdaptiveFusionWeights.m` | 动态权重的核心实现。所有 factorized 权重、PD-weighted GA、FI-weighted GA、FID-FIA baseline、Balanced mode、Cardinality-critical mode 的权重逻辑都汇集在这里。 |
| `multisensorLmb/runParallelUpdateLmbFilter.m` | 动态权重调度入口。它决定何时调用 `computeAdaptiveFusionWeights`，并把输出写回 `model.ga*Weights` / `model.aa*Weights`。 |
| `multisensorLmb/gaLmbTrackMerging.m` | GA/KLA 融合的权重消费端。它分别读取 spatial weights 和 existence weights，决定空间 Gaussian canonical fusion 与 Bernoulli existence fusion。 |
| `multisensorLmb/aaLmbTrackMerging.m` | AA 融合的权重消费端。它读取 AA spatial/existence weights，用于 mixture 拼接和存在概率平均。 |
| `multisensorLmb/runDistributedLmbFilter.m` | 分布式场景包装层。它为每个传感器构造邻域模型、初始 Metropolis/Uniform 权重、结构先验，并调用 `runParallelUpdateLmbFilter`。 |

这五个文件就是“动态权重”本身的主干。其他文件多数是输入生成、实验参数编排或报告统计。

## 调度入口：runParallelUpdateLmbFilter

文件：`multisensorLmb/runParallelUpdateLmbFilter.m`

重点看三段：

1. 初始化 `prevWeights`

```matlab
prevWeights.ga = model.gaSensorWeights;
prevWeights.aa = model.aaSensorWeights;
prevWeights.gaSpatial = getConfigField(model, 'gaSpatialWeights', model.gaSensorWeights);
prevWeights.gaExistence = getConfigField(model, 'gaExistenceWeights', model.gaSensorWeights);
```

这部分定义动态权重的时间递推初值。EMA 平滑、历史项、branch-specific 权重都依赖这些上一时刻权重。

2. 每个传感器先做本地 posterior update

```matlab
measurementUpdatedDistributions{s} = computePosteriorLmbSpatialDistributions(...);
```

动态权重不直接处理 raw measurements，也不直接做 LMB update。它消费的是每个传感器已经更新后的 LMB posterior：`measurementUpdatedDistributions`。

3. 调用动态权重并写回 model

```matlab
[gaWeights, aaWeights, debug] = computeAdaptiveFusionWeights(...);
model.gaSensorWeights = gaWeights;
model.aaSensorWeights = aaWeights;
model.gaSpatialWeights = getConfigField(debug, 'gaSpatialWeights', gaWeights);
model.gaExistenceWeights = getConfigField(debug, 'gaExistenceWeights', gaWeights);
```

这里是动态权重真正接入融合流程的位置。`debug` 里的 branch weights 会覆盖默认 scalar weights；如果有 target-wise weights，也会写入 `model.gaTargetWiseWeights` / `model.aaTargetWiseWeights`。

## 权重核心：computeAdaptiveFusionWeights

文件：`multisensorLmb/computeAdaptiveFusionWeights.m`

这个文件可以按功能块阅读。

### 1. 配置读取

文件开头读取 `model.adaptiveFusion`：

```matlab
method = lower(getField(cfg, 'method', 'factorized'));
useDecoupledKla = getField(cfg, 'useDecoupledKla', false);
useCovariance = getField(cfg, 'useCovariance', true);
useLinkQuality = getField(cfg, 'useLinkQuality', true);
useExistenceConfidence = getField(cfg, 'useExistenceConfidence', false);
useStructureAwareKla = ...
useFidFiaExistence = ...
```

这里决定当前 arm 是普通 factorized 权重，还是 PD/FI/FID-FIA 这类 direct baseline。

### 2. Direct baseline 分支

这三个分支会跳过一般 factor product：

```matlab
isFidFiaMethod(method)
isFiTraceGaMethod(method)
isPdWeightedGaMethod(method)
```

对应实现：

| 函数 | 含义 |
| --- | --- |
| `computePdWeightedGaFusionWeights` | PD-weighted GA。按每个 target/sensor 的检测概率打分，然后汇总成 sensor weights。 |
| `computeFiTraceGaFusionWeights` | FI-weighted GA。用 posterior covariance 的 Fisher trace 近似信息量。 |
| `computeFidFiaFusionWeights` | FID-FIA-weighted GA baseline。用 target pair 的近似 information-geometric distance 累积成传感器分数。 |

这些 direct baseline 返回的权重会同时填入 `gaSpatialWeights`、`gaExistenceWeights`、`aaSpatialWeights`、`aaExistenceWeights`。它们是 baseline，不是当前 Balanced/Cardinality-critical 的 branch-decoupled 主线。

### 3. Factorized backbone

一般动态权重的主分数是：

```matlab
baseScore = availabilityMask .* covScore .* linkQuality;
rawScore = baseScore .* existenceConfidenceScore;
```

当前主线最重要的是：

| 因子 | 代码变量 | 说明 |
| --- | --- | --- |
| posterior concentration | `covScore` | 看本地 posterior 是否集中。 |
| realized communication reliability | `linkQuality` | 看当前传感器信息是否可靠送达。 |
| existence decisiveness | `existenceConfidenceScore` | 看 Bernoulli existence probability 是否果断。 |
| availability | `availabilityMask` | 屏蔽当前不可用或未送达的传感器。 |

`association ambiguity`、`freshness`、`CT-FI decay`、`NIS penalty`、`history` 和 `cardinality-consensus` 已从当前核心函数的主线计算里移除；它们只保留为历史实验和附录讨论材料。

### 4. Decoupled KLA 分支

当 `useDecoupledKla = true` 时，代码不再只输出一条权重，而是拆成 spatial 和 existence 两条路径：

```matlab
spatialDedicatedScore = availabilityMask .* (covScore .^ spatialCovariancePower) .* ...
    (linkQuality .^ spatialLinkQualityPower);

existenceDedicatedScore = availabilityMask .* (linkQuality .^ existenceLinkQualityPower) .* ...
    (existenceConfidenceScore .^ existenceConfidenceWeightPower);
```

然后通过：

```matlab
blendDecoupledScore(...)
finalizeAdaptiveWeights(...)
```

得到：

```matlab
debug.gaSpatialWeights
debug.gaExistenceWeights
debug.aaSpatialWeights
debug.aaExistenceWeights
```

这就是 Balanced mode / Cardinality-critical mode 的关键：空间融合和存在/基数融合不再被同一条 scalar weight 绑死。

### 5. Structure-aware 和 FID-FIA existence refinement

结构相关逻辑主要在这些 helper：

| 函数 | 作用 |
| --- | --- |
| `resolveStructurePrior` | 从 model 读取空间/存在分支的结构先验。 |
| `applyStructurePrior` | 用结构先验调制 score。 |
| `resolveStructureConsistencyScores` | 根据 posterior 结构一致性计算 spatial/existence score。 |
| `computePairwiseStructureDisagreement` | 比较两个本地 posterior 的空间和存在差异。 |

FID-FIA 作为 existence refinement 时，不走 direct baseline，而是由：

```matlab
resolveFidFiaExistenceScore(...)
```

生成 `fidFiaExistenceScore`，再只调制 existence branch：

```matlab
existenceScore = existenceScore .* (fidFiaExistenceScore .^ fidFiaExistenceStrength);
```

这就是 Cardinality-critical mode 与纯 FID-FIA scalar baseline 的主要区别：FID-FIA 信号只进入 existence/cardinality 分支，不替换整个 posterior 的空间权重。

### 6. Score 下界与最终权重下界

这里有两个不同层次的“下界”，不要混在一起理解：

| 类型 | 作用位置 | 目的 |
| --- | --- | --- |
| score 下界 | 某个质量因子从原始诊断量映射到 score 时 | 把某个因子做成软惩罚，避免单个弱因子直接否决传感器。 |
| final weight 下界 | `normalizeScores`、EMA 之后的最终融合权重 | 防止活跃邻居因为瞬时 score 抖动被完全踢出融合。 |

当前核心实现里真正还在使用的 score 下界：

| 配置 | 当前主线含义 | 典型值 | 是否在最终主实验生效 |
| --- | --- | ---: | --- |
| `existenceConfidenceMinScore` | existence confidence 的软下界。`r` 接近 0.5 时只降权，不直接清零。 | `0.85` | Balanced mode 和 Cardinality-critical mode 生效。 |
| `structureReliabilityMinScore` | 通信可靠性 prior 的软下界，避免高丢包节点在结构先验里被完全消除。 | `0.25` | Balanced mode 和 Cardinality-critical mode 生效。 |
| `spatialConsistencyMinScore` / `existenceConsistencyMinScore` | posterior structure consistency 分支的软下界。 | `0.4` | 当前主实验 `usePosteriorStructureConsistency=false`，不生效。 |
| `fidFiaExistenceMinScore` | FID-FIA existence refinement 的软/硬抑制边界。 | 默认 `0.4`，最终 arm 为 `0.0` | Cardinality-critical mode 生效，且设为 `0.0`。 |

当前 final weight 下界：

| 配置 | 当前主线含义 | 典型值 | 说明 |
| --- | --- | ---: | --- |
| `minWeight` | 非解耦 scalar 权重路径的最终权重下界。 | `0.05` | factorized baseline 使用；direct FID-FIA baseline 主实验设 `fidFiaMinWeight=0.0`。 |
| `spatialMinWeight` | spatial branch 最终权重下界。 | `0.05` | 保留空间融合的邻居多样性，避免定位分支过早塌缩。 |
| `existenceMinWeight` | existence branch 最终权重下界。 | Balanced 为 `0.05`，Cardinality-critical 为 `0.0` | Cardinality-critical 允许 FID-FIA 对 existence/cardinality 分支做强抑制。 |

为什么这些设置是合理的：

- `covScore` 不设 score 下界：空 posterior 或没有有效 Gaussian component 的节点应当自然降到 0。
- `linkQuality` 不设 score 下界：当前步信息全部丢失时，应当明显降权；硬可用性由 `availabilityMask` 控制。
- `existenceConfidenceMinScore=0.85` 是温和校正：存在性不果断不等于空间估计完全无用，所以只轻量降权。
- `spatialMinWeight=0.05` 在 8-sensor 主实验里最多占 `8 * 0.05 = 0.4` 的 floor budget，仍给动态分数留下足够自由度。
- `fidFiaExistenceMinScore=0.0` 和 `existenceMinWeight=0.0` 是 Cardinality-critical 的关键：它不是一般稳定化设置，而是为了让 FID-FIA 在目标数风险高时可以真正压低不可靠 existence 分支。

### 7. 归一化、平滑和 debug

最后几个 helper 是动态权重稳定性的核心：

| 函数 | 作用 |
| --- | --- |
| `normalizeScores` | 把非负 score 转为权重。 |
| `enforceMinimumWeight` | 保底权重，避免单个节点被完全饿死。 |
| `finalizeAdaptiveWeights` | 对 score 做归一化、EMA 平滑、min-weight 处理。 |
| `buildDirectWeightDebug` | direct baseline 的 debug 字段补齐。 |

`debug` 结构很重要。它不是附属品，实验报告和排错都依赖它区分到底是 `covScore`、`linkQuality`、`existenceConfidenceScore`、`fidFiaExistenceScore` 还是结构项在驱动权重。

## 权重消费端：GA 和 AA merging

### GA/KLA：gaLmbTrackMerging

文件：`multisensorLmb/gaLmbTrackMerging.m`

核心读取：

```matlab
spatialWeights = resolveObjectWeightVector(model, 'gaTargetWiseWeights', ...
    'gaSpatialWeights', model.gaSensorWeights, i);
existenceWeights = resolveObjectWeightVector(model, 'gaTargetWiseWeights', ...
    'gaExistenceWeights', model.gaSensorWeights, i);
```

空间分支用 `spatialWeights` 加权 Gaussian canonical parameters：

```matlab
KMatched = spatialWeights(s) * inv(T);
```

存在分支用 `existenceWeights` 加权 Bernoulli existence：

```matlab
numerator = numerator * (rS^(existenceWeights(s)));
partialDenominator = partialDenominator * ((1-rS)^(existenceWeights(s)));
```

所以如果你改了 `computeAdaptiveFusionWeights` 的 branch outputs，最终行为变化会在这里落地。

### AA：aaLmbTrackMerging

文件：`multisensorLmb/aaLmbTrackMerging.m`

AA 也读取 branch weights：

```matlab
spatialWeights = resolveWeightVector(model, 'aaSpatialWeights', model.aaSensorWeights);
existenceWeights = resolveWeightVector(model, 'aaExistenceWeights', model.aaSensorWeights);
```

区别是 AA 用 spatial weights 加权并拼接 Gaussian mixture，用 existence weights 做线性加权。当前论文主线主要是 GA/KLA，但 AA 路径保留了动态权重兼容性。

## 分布式包装：runDistributedLmbFilter

文件：`multisensorLmb/runDistributedLmbFilter.m`

这个文件不直接算动态权重，但它决定分布式场景里每个节点看到什么邻域、初始权重和结构先验。

重点看：

```matlab
weightsBySensor = computeMetropolisWeights(neighborMap);
localModels{s} = buildSubModel(model, neighborIdx);
localModels{s}.gaSensorWeights = weightsBySensor{s};
localModels{s}.gaSpatialWeights = weightsBySensor{s};
localModels{s}.gaExistenceWeights = weightsBySensor{s};
```

以及：

```matlab
[spatialStructurePrior, existenceStructurePrior] = computeLocalStructurePriors(...);
localModels{s}.gaSpatialStructurePrior = spatialStructurePrior;
localModels{s}.gaExistenceStructurePrior = existenceStructurePrior;
```

如果要改“拓扑/邻域如何影响动态权重”，优先改这里的 `computeLocalStructurePriors`，而不是去改 fusion merge 端。

## 辅助输入文件

下面这些文件不是动态权重主体，但会给动态权重提供诊断输入。

| 文件 | 何时需要看 |
| --- | --- |
| `multisensorLmb/generateLmbSensorAssociationMatrices.m` | 只在你关心关联矩阵、NIS 诊断或 association ambiguity 诊断时需要看。当前动态权重核心不再消费这些诊断量。 |
| `multisensorLmb/puLmbTrackMerging.m` | PU baseline，不消费动态 GA/AA 权重；只在比较 PU/GA/AA 融合规则时看。 |

## 实验配置入口

动态权重实验的主要配置不在核心函数里硬编码，而在 `RUN/GA` 的实验脚本里。

| 文件 | 角色 |
| --- | --- |
| `RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m` | 最重要的动态权重消融入口。这里构造 fixed、covariance、link quality、existence confidence、Balanced、Cardinality-critical、PD/FI/FID-FIA 等 arms。 |
| `RUN/GA/runMultisensorFilters_formation_4plus4_CommLevelThreeMethodCompare.m` | 通信等级对照入口，比较 Fixed Metropolis、Balanced mode、Cardinality-critical mode。 |
| `RUN/GA/runMultisensorFilters_formation_4plus4_IdealCommCompare.m` | ideal communication 支撑实验入口，用于确认动态权重不是只在丢包时有效。 |
| `RUN/GA/runMultisensorFilters_formation_4plus4_StateDependentQualityFalseTargetsCompare.m` | state-dependent quality / false-target 支撑实验入口。 |
| `RUN/GA/runMultisensorFilters_formation_4plus4_NISCompare.m`、`HistoryCompare.m`、`TieredLinkFreshnessCompare.m` | 历史次线模块对照入口；当前主线代码已经不再从 `computeAdaptiveFusionWeights.m` 启用这些分数。 |

## 如果要改代码，应该改哪里

| 目标 | 优先改动位置 |
| --- | --- |
| 加一个新的动态权重因子 | `computeAdaptiveFusionWeights.m`：配置读取、factor helper、`rawScore` 或 branch score、`debug` 字段。 |
| 加一个新的 direct baseline | `computeAdaptiveFusionWeights.m`：新增 `isXMethod` 和 `computeXFusionWeights`，再在 `RUN/GA/*Ablation.m` 增加 arm。 |
| 改 Balanced mode | `RUN/GA/*Ablation.m` 的 arm 配置，以及 `computeAdaptiveFusionWeights.m` 的 decoupled/structure 参数解释。 |
| 改 Cardinality-critical mode | `computeAdaptiveFusionWeights.m` 的 `useFidFiaExistence` 分支和 existence branch 参数；实验配置在 `RUN/GA/*Ablation.m`。 |
| 改空间权重和存在权重如何落到融合公式里 | `gaLmbTrackMerging.m`；AA 路径则看 `aaLmbTrackMerging.m`。 |
| 改分布式拓扑/邻居对权重的影响 | `runDistributedLmbFilter.m` 的 `computeNeighborMap`、`computeLocalStructurePriors`、`computeMetropolisWeights`。 |
| 排查权重为什么异常 | 先看 `runParallelUpdateLmbFilter.m` 写回的 `debug` 字段，再看 `computeAdaptiveFusionWeights.m` 的 `availabilityMask`、`rawScore`、`spatialRawScore`、`existenceRawScore`。 |

## 暂不作为动态权重主线的部分

- `common/`：本次暂不展开。它们提供测量、通信统计、传感器质量、评估指标等输入，但不是动态权重的主体。
- `lmb/`、`lmbm/`、`multisensorLmbm/`：这些是滤波和 LMBM 参考实现，不是动态 GA/AA 权重主线。
- `RUN/GA/*.md`：实验报告产物，不是实现代码；干净代码包中已不保留。
