# Adaptive Fusion Weights (GA/AA)

本文档说明当前工程中的自适应融合权重结构、推荐主线配置，以及哪些模块应放在正文主线、哪些模块只保留为次线或附录材料。

当前建议已经比较明确：

- 正文主线只突出 `covariance + link quality + existenceConfidence + weak structure-aware decoupled KLA`
- `NIS/history/freshness` 只保留历史实验结果和附录讨论价值，当前核心权重函数不再计算这些分数

## 1. 当前主线结构

当前 `main` 分支上的 best 配置，实际使用的是四层主线结构：

```text
baseScore_j(t) = mask_j(t) * covScore_j(t) * linkQuality_j(t)
rawScore_j(t)  = baseScore_j(t) * existenceConfidence_j(t)
```

在此基础上，再做一个很弱的 decoupled refinement：

```text
spatialScore_j(t)   = blend(rawScore_j(t), spatialDedicatedScore_j(t), eta_s)
existenceScore_j(t) = blend(rawScore_j(t), existenceDedicatedScore_j(t), eta_e)
```

如果进一步启用 structure-aware refinement，则只做轻量调制：

```text
spatialScore_j(t)   = spatialScore_j(t)   * spatialStructurePrior_j(t)^gamma_s
existenceScore_j(t) = existenceScore_j(t) * existenceStructurePrior_j(t)^gamma_e
```

当前 best 对应的主线开关是：

```text
useCovariance = true
useLinkQuality = true
useExistenceConfidence = true
useDecoupledKla = true
useStructureAwareKla = true
% NIS/history/freshness 等次线分数不再进入当前核心权重函数
```

当前实现已经把 `innovationPenalty`、`historyScore`、`freshnessScore` 等次线分数从 `computeAdaptiveFusionWeights.m` 中清理掉，避免正文主线和历史尝试混在同一个核心函数里。

## 2. 主线四项的具体实现与作用

### 2.1 协方差质量项 `covScore`

实现位置：

- `multisensorLmb/computeAdaptiveFusionWeights.m`

当前实现对 measurement-updated LMB 做 m-projection 后，用 posterior covariance trace 构造质量分数：

```text
covScore = 1 / (eps + mean(trace(T)))
```

它表达的是“这个局部后验是否足够集中、状态估计是否足够精确”。

它有用的原因是：

- fixed weights 默认假设各节点后验质量接近，但实际局部估计精度会明显波动
- `covScore` 直接给状态更集中的后验更高权重，是最自然的 posterior-quality baseline
- 在当前 tiered-drop 主场景中，它是 fixed weights 之后第一个稳定拉低三项共识误差指标的主因子

### 2.2 链路质量项 `linkQuality`

实现位置：

- `multisensorLmb/computeAdaptiveFusionWeights.m`

当前实现直接基于通信层 delivered / dropped 统计：

```text
linkQuality = delivered / (delivered + dropped)
```

它表达的是“这个节点最近真正把多少有效测量送出来了”。

它有用的原因是：

- 协方差只能看 posterior 质量，看不到通信是否可靠
- 在分档异构丢包条件下，不同节点的长期送达率确实不同
- 该因子能够把“好后验但链路差”和“后验一般但链路稳定”区分开，是当前 heterogeneous communication 设定下最关键的通信侧因子

### 2.3 存在性/基数置信度项 `existenceConfidence`

实现位置：

- `multisensorLmb/computeAdaptiveFusionWeights.m`

这个因子不再看状态协方差，也不再看链路是否送达，而是直接看 measurement-updated posterior 中 Bernoulli existence probability `r` 是否足够尖锐。

对单个 Bernoulli：

```text
certainty = |2r - 1|
```

对单个传感器的汇总实现是：

```text
weightedConfidence = sum(r .* |2r - 1|) / sum(r)
existenceConfidenceScore = minScore + (1 - minScore) * weightedConfidence^power
```

它有用的原因是：

- `covScore` 只能表达“位置是否集中”，不能表达“存在/不存在判决是否果断”
- `linkQuality` 只能表达“有没有顺利发过来”，不能表达“传来的 posterior 在 cardinality 上是否可信”
- `existenceConfidence` 恰好补上了这条维度，因此能更直接对应 cardinality dispersion

在当前 tiered-drop `5-trial` 消融里：

```text
+link quality:         OSPA 1.877771, RMSE 1.800945, Card 0.245250
+existence confidence: OSPA 1.874840, RMSE 1.779820, Card 0.244500
```

它是当前已验证最有效的第三个主线因子。

### 2.4 弱结构先验解耦项 `structure-aware decoupled KLA`

实现位置：

- `multisensorLmb/computeAdaptiveFusionWeights.m`
- `multisensorLmb/runDistributedLmbFilter.m`

这一层不是替代前三个质量因子的新主分数，而是在三因子 baseline 上做“很弱的分支 refinement”。

当前实现里：

- `spatial` 分支可以承受相对更强的结构修正
- `existence` 分支只允许非常轻的结构调制
- 结构先验来自局部子图重叠和链路可靠性信息

它有用的原因是：

- spatial consistency 和 existence consistency 对权重扰动的敏感度不同
- 如果两者完全共用一条权重路径，往往会出现“空间更好了，但 cardinality 被拖坏”的耦合
- decoupled KLA 先把两条分支拆开，再在 spatial 分支上保留主要结构增益，只给 existence 分支非常轻的修正，能更稳定地换取 OSPA/RMSE 改善而不破坏 cardinality

当前 best `5-trial` 结果是：

```text
+existence confidence baseline:   OSPA 1.874840, RMSE 1.779820, Card 0.244500
+structure-aware decoupled KLA:   OSPA 1.862244, RMSE 1.749608, Card 0.244250
```

这说明它更适合作为“主线三因子之上的轻量 refinement”，而不是单独拿出来当一个拓扑权重方法。

## 3. 当前推荐配置

### 3.1 当前主线 best 配置

当前推荐直接使用以下配置复现主线 best：

```matlab
model.adaptiveFusion.enabled = true;
model.adaptiveFusion.emaAlpha = 0.7;
model.adaptiveFusion.minWeight = 0.05;

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
model.adaptiveFusion.structureReliabilityMinScore = 0.25;
```

clean 版本不保留生成报告；需要复现时从
`RUN/GA/runMultisensorFilters_formation_4plus4_TieredLinkAblation.m`
重新生成。

对应的 main-line headline numbers：

```text
fixed weights:                    OSPA 2.624065, RMSE 2.702602, Card 0.878750
+covariance:                      OSPA 2.211513, RMSE 2.410976, Card 0.589500
+link quality:                    OSPA 1.877771, RMSE 1.800945, Card 0.245250
+existence confidence:            OSPA 1.874840, RMSE 1.779820, Card 0.244500
+structure-aware decoupled KLA:   OSPA 1.862244, RMSE 1.749608, Card 0.244250
```

### 3.2 三因子 baseline 配置

如果只想回到结构 refinement 之前的三因子 baseline，可使用：

```matlab
model.adaptiveFusion.useCovariance = true;
model.adaptiveFusion.useLinkQuality = true;
model.adaptiveFusion.useExistenceConfidence = true;
model.adaptiveFusion.existenceConfidenceMinScore = 0.85;
model.adaptiveFusion.existenceConfidencePower = 2.0;
model.adaptiveFusion.useDecoupledKla = false;
model.adaptiveFusion.useStructureAwareKla = false;
```

clean 版本不保留对应的历史报告；需要检查该配置时从当前 GA ablation
入口重新运行。

## 4. 次线模块与附录候选

### 4.1 `robust NIS`

实现位置：

- `multisensorLmb/generateLmbSensorAssociationMatrices.m`

当前定位：

- 作为一致性惩罚或统计诊断模块保留
- 不作为当前正文主线的一部分

原因：

- NIS 与 posterior covariance 通过 innovation covariance 有天然耦合
- 如果把它再当成一个“越小越好”的奖励项，容易和 `covScore` 重复表达
- 当前更合理的做法是把它当一致性测试，只在明显失配时惩罚

当前保留的次线结果是：

```text
w/o NIS -> robust NIS -> NIS
OSPA: 1.811 -> 1.810 -> 1.901
RMSE: 3.173 -> 3.153 -> 3.329
Card: 0.214 -> 0.209 -> 0.234
```

结论：

- `robust NIS` 明显好于 plain `NIS`
- 但它相对 `w/o NIS` 的增益很弱，不足以压过当前四项主线

clean 版本不保留 NIS 对应的历史报告或单独 GA 入口。

### 4.2 `historyScore`

历史实现位置：

- 曾位于 `multisensorLmb/computeAdaptiveFusionWeights.m`，当前主线版本已清理

当前定位：

- 只保留为次线或附录材料
- 当前核心实现不再提供开关式启用路径

原因：

- 它会引入额外的时间耦合路径，不利于判断当前主线因子的真实边际收益
- 当前实验里收益不稳定，且 narrative 上不如 `existenceConfidence` 和弱结构解耦清晰

当前保留的历史结果是：

```text
w/o history -> history
OSPA: 1.811 -> 1.814
RMSE: 3.173 -> 3.158
Card: 0.214 -> 0.215
```

结论：

- history 只带来很弱的局部变化
- 不适合放在正文主线，最多可在附录里说明“尝试过，但收益有限且耦合较强”

clean 版本不保留 history 对应的历史报告或单独 GA 入口。

## 5. 回滚方式

如果需要回退到三因子 existence-confidence baseline，可关闭当前弱结构解耦层：

```matlab
model.adaptiveFusion.useDecoupledKla = false;
model.adaptiveFusion.useStructureAwareKla = false;
```

如果需要重新做 NIS/history/freshness 次线分析，应从对应历史提交或实验分支恢复实现，再单独作为附录实验入口维护；当前 `computeAdaptiveFusionWeights.m` 不再通过配置开关启用这些分数。

## 6. 当前实验建议

当前更建议优先围绕下面这条主线补证据：

1. `fixed -> +covariance -> +link quality -> +existence confidence -> +structure-aware decoupled KLA`
2. 在更多 seed 和更多通信等级下验证弱结构 refinement 是否稳定
3. 只把 `NIS/history/freshness` 作为历史次线或附录对照，不再把它们放在正文主故事的前排
