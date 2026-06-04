# 通信层仿真方案（多级）

本文档描述多传感器仿真中的通信层模拟规则，用于在**测量进入滤波器之前**引入带宽限制、链路干扰与节点失联。实现目标是：**不改核心滤波算法，只改变 measurements 的交付方式**。

## 1. 基本假设

- 量测是**周期性同步**产生的：同一时刻 `t` 有 `measurements{s, t}`。
- 每个传感器在每个周期产生一个量测集合：`measurements{s, t}`。
- 链路丢包与失联以“传感器包”为粒度（传感器在该周期的量测集合），带宽限制可截断该集合。

## 2. 多级通信环境定义

### Level 0：理想通信

- 不进行任何处理，`measurements` 原样透传。

### Level 1：全局通信量上限（按量测数计数）

每个时间步有一个**全局上限** `globalMaxMeasurementsPerStep`，最多允许这么多量测上传。

规则：
- **优先级由权重决定**：权重高的传感器先分配预算。
- 预算不足时，对当前传感器的量测集合进行截断（按配置选择保留前 K 条或随机 K 条）。
- 空集合不消耗预算。

推荐排序策略（可配置）：
- `weightedPriority`：按权重从高到低排序；若权重相同，可按传感器编号或随机打散（避免固定偏置）。

### Level 2：链路干扰（先限带宽，再丢包）

在 Level 1 的结果上引入链路丢包：
- 对每个已允许上传的传感器包，按丢包模型决定是否丢弃。
- 丢包模型可选：
  - 固定丢包率 `pDrop`
  - 按传感器显式指定的丢包率向量 `pDropBySensor`
  - 分档丢包配置 `pDropLevels + pDropLevelCounts`
  - 两状态马尔可夫链路（Good/Bad 状态，分别有不同丢包率）

**时延**：暂不实现，但预留接口（后续可加入“延迟队列 + 到达时间戳”）。

### Level 3：节点失联

在 Level 2 基础上引入节点失联：
- 最多允许 `maxOutageNodes = 1` 个传感器在某个时间段**完全断开**。
- 失联期间该传感器的包全部丢弃（不参与带宽排序或视为失联后直接清空）。
- 失联时间窗可以固定或随机生成。

## 3. 处理流程（伪流程）

以时间步 `t` 为单位：
1. 收集所有传感器的量测集合：`measurements{s, t}`。
2. Level 1：按权重排序并消耗量测预算（`globalMaxMeasurementsPerStep`），必要时截断集合。
3. Level 2：对已允许上传的传感器集合执行丢包判定。
4. Level 3：若传感器在失联窗口，直接丢弃包。
5. 输出 `measurementsDelivered{s, t}`。

> 应用顺序：**先带宽限制，再链路丢包，再节点失联**（失联可等价为丢包率 1 的特殊状态）。

## 4. 配置结构建议（commConfig）

```matlab
commConfig.level = 2; % 0/1/2/3

% Level 1: global budget
commConfig.globalMaxMeasurementsPerStep = 20;
commConfig.sensorWeights = [0.6, 0.3, 0.1]; % 与传感器数量一致
commConfig.priorityPolicy = 'weightedPriority'; % weightedPriority / weightedShuffle
commConfig.measurementSelectionPolicy = 'firstK'; % firstK / random

% Level 2: link loss
commConfig.linkModel = 'fixed'; % fixed / markov
commConfig.pDrop = 0.2;         % fixed mode
commConfig.pDropBySensor = [];  % 可选：长度=传感器数，优先级高于 pDrop
commConfig.pDropLevels = [];    % 可选：例如 [0, 0.1, 0.2, 0.5]
commConfig.pDropLevelCounts = []; % 可选：例如 [1, 4, 1, 2]
commConfig.pGoodToBad = 0.1;    % markov mode
commConfig.pBadToGood = 0.3;
commConfig.pDropGood = 0.05;
commConfig.pDropBad = 0.4;

% Level 3: outage
commConfig.maxOutageNodes = 1;
commConfig.outageSchedule = []; % 空=随机生成
commConfig.outageMinDuration = 10;
commConfig.outageMaxDuration = 30;
```

### 字段说明（简表）

- `level`：通信环境等级（0/1/2/3）
- `globalMaxMeasurementsPerStep`：每步全局允许上传的量测数上限
- `sensorWeights`：传感器优先级权重（长度需与传感器数一致）
- `priorityPolicy`：排序策略（`weightedPriority` 按权重排序；`weightedShuffle` 权重相同的随机打散）
- `measurementSelectionPolicy`：量测截断策略（`firstK` 保留前 K 条；`random` 随机 K 条）
- `linkModel`：链路模型（`fixed` 固定丢包率；`markov` 两状态链路）
- `pDrop`：固定丢包率（`linkModel='fixed'` 时生效）
- `pDropBySensor`：按传感器指定的固定丢包率向量；若提供，则优先于标量 `pDrop`
- `pDropLevels`：分档丢包率列表，例如 `[0, 0.1, 0.2, 0.5]`
- `pDropLevelCounts`：每个档位对应的传感器个数；与 `pDropLevels` 等长，且总和必须等于传感器数
- `pGoodToBad` / `pBadToGood`：链路状态转移概率（`markov` 模式）
- `pDropGood` / `pDropBad`：Good/Bad 状态下的丢包率（`markov` 模式）
- `maxOutageNodes`：最多失联传感器数量（Level 3）
- `outageSchedule`：失联时段计划（空则随机生成）
- `outageMinDuration` / `outageMaxDuration`：随机失联时长范围（步数）

> 兼容说明：若历史配置仍使用 `globalMaxSensorPacketsPerStep`，系统会将其视为“量测数上限”处理。

### Level 2 的优先级规则

当 `linkModel='fixed'` 时，链路丢包率的解析优先级为：

1. `pDropBySensor`
2. `pDropLevels + pDropLevelCounts`
3. 标量 `pDrop`

也就是说：

- 如果直接给了 `pDropBySensor`，则每个传感器使用该向量中对应的固定丢包率。
- 如果没有给 `pDropBySensor`，但给了 `pDropLevels + pDropLevelCounts`，系统会先按档位生成一个长度等于传感器数的 `pDropBySensor`，再在 trial 内固定使用。
- 如果上面两者都没给，才退回到统一标量 `pDrop`。

### 分档丢包的推荐写法

如果希望链路质量更接近“好/中/差/极差”的离散层级，而不是所有节点共享同一个 `pDrop`，推荐使用：

```matlab
commConfig.level = 2;
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2; % 目标平均值，仅用于记录和回退
commConfig.pDropLevels = [0, 0.1, 0.2, 0.5];
commConfig.pDropLevelCounts = [1, 4, 1, 2];
```

该配置的含义是：

- 1 个节点处于 `0` 档
- 4 个节点处于 `0.1` 档
- 1 个节点处于 `0.2` 档
- 2 个节点处于 `0.5` 档

对于 8 传感器场景，上述组合的总体平均丢包率恰好为：

$$
\frac{1 \times 0 + 4 \times 0.1 + 1 \times 0.2 + 2 \times 0.5}{8} = 0.2
$$

因此它可以在**保持整体通信强度与旧配置一致**的前提下，引入更强的节点间异构性。

## 5. 统计信息（建议输出）

通信层可输出 `commStats` 以便分析：
- 每步允许上传的传感器数量
- 每传感器丢包率
- 失联传感器与失联窗口
- 有效上传量测数随时间变化

## 6. 集成位置建议

通信层作为预处理函数调用，入口脚本中插入：

```matlab
[measurementsDelivered, commStats] = applyCommunicationModel(measurements, model, commConfig);
stateEstimates = runParallelUpdateLmbFilter(model, measurementsDelivered);
```

### 推荐配置（稳定版）

以下配置为当前测试较稳定的 Level 1 方案，可直接用于 `runMultisensorFilters.m`：

```matlab
commConfig = struct();
commConfig.level = 1; % 0=ideal, 1=bandwidth, 2=link loss, 3=node outage
commConfig.globalMaxMeasurementsPerStep = 35;
commConfig.sensorWeights = ones(1, numberOfSensors) / numberOfSensors;
commConfig.priorityPolicy = 'weightedPriority';
commConfig.measurementSelectionPolicy = 'firstK';
commConfig.linkModel = 'fixed';
commConfig.pDrop = 0.2;
commConfig.maxOutageNodes = 1;
```

### 推荐配置（分档异构链路版）

以下配置用于当前 formation `4+4` 实验，目标是保留平均 `pDrop=0.2`，同时让不同节点处于不同档位：

```matlab
commConfig = struct();
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

这套配置的优点是：

- 总体平均丢包率不变，方便与旧实验对齐
- 节点链路质量分层明确，解释性更强
- `linkQuality` 在自适应权重中的区分度更高

推荐集成点：
- `runMultisensorFilters.m`
- `runAdvancedPerformanceAnalysis.m`（如需评估通信影响）

## 7. 预留扩展（时延）

后续引入时延时，可在通信层加入：
- `delayQueue`：缓存未到达包
- `arrivalTime`：明确延迟步数
- 可选 “迟到即丢弃” 或 “迟到仍可用” 策略

当前版本暂不实现时延，仅保留接口与配置占位。
