# HanBayes 算法细节

本文档描述 HanBayes 模型族（StandardNB / FWNB / DFWNB-v2 / SDFWNB）的完整算法流程，与 `configs/frozen.json` 中的冻结超参数一一对应。

## 1. 符号与数据协议

- 二分类：`y ∈ {0(负面), 1(正面)}`；
- 特征：规范化文本上的字符 N-gram（冻结配置为 `ngram_range=(2,2)`，即字符二元组）；
- 词表：仅由训练集构建，按 `(DF↓, TF↓, 字典序↑)` 排序后截取前 `max_features=50000` 个，`min_df=2`；
- 最终测试协议：train+dev 合并（8682 条）为最终训练集，test（1178 条）一次性评估；
- **架构约定**：整个模型族共享同一次文本扫描产出的 CSR 稀疏计数矩阵 `C ∈ ℕ^{N×V}`，训练与评分全部为向量化稀疏线性代数。

### 文本规范化（`text.py`）

1. HTML 剥离、NFKC 规范化、转小写；
2. 连续标点折叠为单标点，数字/网址/邮箱替换为占位符（`数字`/`网址`/`邮箱`）；
3. 纯标点 N-gram 跳过；
4. 行首行尾仅剥离 `，。；：、`（忠实于原始实验设定）。

### 特征提取（`features.py`）

`CharNgramVectorizer` 是**唯一**扫描原始文本的组件：

- `fit`：统计 DF/TF → `min_df` 过滤 → `(DF↓, TF↓, 字典序↑)` 排序 → 截取 `max_features`；
- `transform`：每篇文本一次扫描产出 CSR 计数行，行内索引按规范顺序排列，保证确定性。

## 2. StandardNB（基线）

多项式朴素贝叶斯 + 拉普拉斯平滑（`alpha=0.05`）：

```
log P(c) + Σ_f n_f(x) · log P(f|c)，  P(f|c) = (N_cf + α) / (N_c + αV)
```

向量化形式：`scores = log_prior + C @ log_prob.T`。

## 3. FWNB：互信息判别性加权

**动机**：朴素贝叶斯把所有特征一视同仁，但「很满」「意」这类弱信息特征与「满意」这类强信息特征的证据权重应当不同。

1. 从二值存在矩阵计算每个特征与类别的**互信息**（nat）：
   `C` 的非零项置 1 得到存在矩阵 `B`，按类别汇总得 `class_df[2×V]`；
2. MI → 权重映射（`minimum_weight=0.25, maximum_boost=0.5, gamma=1.0`）：

```
w(f) = 0.25 + (1.5 − 0.25) · (MI(f) / MI_max)^γ
```

3. 评分：`scores = log_prior + C @ (w ⊙ log_prob).T`。

**退化性质**：`w ≡ 1` 时严格等价于 StandardNB（有单元测试保证）。

## 4. DFWNB-v2：文档级冗余抑制

**动机**：语义重复的特征（如「满意」「很满意」）会重复计票，放大同一证据。

### 4.1 候选与共现统计

- 按 MI 取前 `candidate_feature_count=2000` 个**纯中文**二元特征为候选；
- 每篇文档保留最多 `maximum_features_per_document=80` 个候选（按 MI 降序），两两组成无序对，统计**文档级共现频率** `pair_df(a,b)`。

### 4.2 非对称冗余得分

对每个共现对 `(a,b)`（`pair_df ≥ minimum_pair_df=20`）依次检查：

1. **共享字过滤**：`have_shared_character(a,b)` 为真则跳过（相邻二元组的重叠是伪冗余）；
2. **方向一致性**：`direction(a)·direction(b) ≤ 0` 则跳过。方向分 `direction(f) = logit(P(f|正面)) − logit(P(f|负面))`（平滑 0.5）；
3. **Jaccard**：`J = pair_df / (df_a + df_b − pair_df) ≥ minimum_jaccard=0.03`；
4. 关系强度：`relation = J × 方向相似度 × 支持度因子`，其中方向相似度 `= min(|d_a|,|d_b|)/max(|d_a|,|d_b|)`，支持度因子 `= pair_df/(pair_df + support_tau)`，`support_tau=30`；
5. **惩罚方向**：MI 较小者被惩罚（MI 相同则 DF 较小者被惩罚）。

每个被惩罚特征的冗余得分取其前 `top_k=3` 强关系的均值，全库按 95% 分位数归一化并截断到 `[0,1]`。

### 4.3 冗余感知权重

```
w'(f) = w(f) / (1 + strength · score(f)^γ)，  strength=1.0, γ=0.5
```

随后做**均值保持**缩放（`w'.mean() = w.mean()`），避免整体权重漂移。

**退化性质**：`score ≡ 0` 时 `w' = w`，严格退化为 FWNB。

## 5. SDFWNB：稀疏类别条件局部依赖修正

**动机**：条件独立性假设在「不满→满意」这类相邻对上系统性失效——这两个 bigram 从不同时独立出现。

### 5.1 有序相邻对

滑动三元组窗口 `(c₁,c₂,c₃)` 产生有序对 `(c₁c₂, c₂c₃)`，要求两侧均为**纯中文**二元组、窗口不含占位符、两侧特征均在词表内且不等。每篇文档中每对最多计一次。

`build_pair_matrix` 返回 `(稀疏存在矩阵 P, DependencyPairRegistry)`，registry 维护 pair ↔ 列的双向映射，使评估集可以重映射到训练列空间。

### 5.2 类别条件依赖量

对满足 `total_df ≥ minimum_pair_df=8` 的对，按类别估计：

```
Δ_c = clip( log[ (pair_df_c + s) / (N_c + 2s) ]
          − log[ (left_df_c + s) / (N_c + 2s) ]
          − log[ (right_df_c + s) / (N_c + 2s) ], ±delta_clip )
```

其中 `s=0.5` 为平滑，`delta_clip=4.0`。随后：

1. **支持度收缩**：`Δ_c ×= total_df / (total_df + support_tau)`（`support_tau=30`）；
2. **跨类别中心化**：`Δ_c −= mean_c(Δ_c)`（只保留类别间的对比信息）；
3. **选择**：按 `|Δ_1 − Δ_0|` 降序取前 `dependency_pair_count=1000` 对，构建修正字典。

### 5.3 评分

对文本 `x` 中出现的每对（每文档最多 `maximum_pairs_per_document=30` 对，按相关性降序截断）：

```
scores(x) = weighted_NB_scores(x) + strength · Σ_pair Δ_c(pair)，  strength=1.0
```

向量化实现：`scores += strength · (P_capped @ correction_matrix)`。

**退化性质**：`strength = 0` 时严格退化为 DFWNB-v2。

## 6. 精确可解释性

整个模型族对特征证据**线性**，因此每个预测可精确分解：

```
logit_c(x) = log_prior[c]
           + Σ_f count(f) · w(f) · log P(f|c)      （特征证据）
           + Σ_pair strength · Δ_c(pair)           （依赖修正，仅 SDFWNB）
```

`ExplanationEngine` 按此式逐项还原，无任何近似。全局解释表（`top_features`、`top_dependencies`）与局部解释（`explain_prediction`）共用同一套数字——**解释与预测严格一致**，这是黑盒模型无法做到的。

## 7. 评估与统计检验

- 指标：Accuracy / Precision / Recall / F1（正、负类）/ Macro / Balanced Accuracy / AUC（无 sklearn 实现）；
- **McNemar 精确检验**：模型间预测差异显著性；
- **配对 Bootstrap**：Macro-F1 差值的 95% CI（5000 次重采样，种子 142）。

## 8. 冻结数字（最终测试协议）

| 模型 | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| StandardNB | 0.7793 | 0.7786 | 0.8505 |
| FWNB | 0.8022 | 0.8009 | 0.8770 |
| DFWNB-v2 | 0.8048 | 0.8033 | 0.8822 |
| **SDFWNB** | **0.8073** | **0.8065** | **0.8867** |

- 标准NB → SDFWNB：McNemar 精确检验 p = 0.000038（显著）；
- 标准NB → SDFWNB：Macro-F1 差值 95% CI [0.0153, 0.0409]；
- FWNB → DFWNB-v2 → SDFWNB：渐进式改进（配对检验不显著，属预期的渐进设计）。

## 9. 实现架构备注

- **单次扫描**：`CharNgramVectorizer.fit_transform` 只扫一次语料；评估时 `transform` 复用词表；
- **单一稀疏矩阵**：四个模型共享同一 CSR 计数矩阵与类别统计（`class_df`、MI 只算一次）；
- **依赖对列映射**：`DependencyPairRegistry` 让评估文本的 pair 矩阵列对齐训练空间——评估集中未出现在训练词表的对**直接丢弃**（它们没有学到的修正量），不产生任何隐式映射；
- **确定性**：词表排序、pair 排序、表格排序全部有确定键，任何机器上重跑结果逐位一致。
