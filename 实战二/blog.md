# 🚀 实战二：信用卡欺诈检测 - 应对极端不平衡数据

> **作者**：渊鱼986 | **发布时间**：2025-07-23 | **阅读时长**：约 20 分钟
> **适合人群**：机器学习进阶者 | **难度等级**：⭐⭐⭐


## 📖 前言

信用卡欺诈每年给全球金融机构和消费者带来数十亿美元的损失。如何从海量交易数据中精准、快速地识别出欺诈行为，是金融风控领域的核心挑战。本项目将带你深入探讨在**极端不平衡数据**场景下的欺诈检测建模，这是一个非常典型且有价值的实战案例。

### 🎯 你将学到什么

- ✅ 处理高度不平衡数据集的策略 (超过 99% 的样本为正常交易)
- ✅ 处理高度不平衡数据集的策略 (超过 99% 的样本为正常交易)
- ✅ **模型对比**：`Logistic Regression` vs `XGBoost` vs `Isolation Forest`
- ✅ **消融实验**：对比有无类别权重对 `XGBoost` 性能的影响
- ✅ 关键评估指标的选择：`AUPRC` (Area Under Precision-Recall Curve) 的重要性
- ✅ 如何在保证高召回率（Recall）的同时，兼顾准确率（Precision）

---

## 🔍 第一步：数据加载与特殊性分析

### 数据集介绍

我们使用 Kaggle 上广受欢迎的“Credit Card Fraud Detection”数据集。该数据集包含了由欧洲持卡人于2013年9月通过信用卡进行的交易。为了保护隐私，数据集中的大部分特征（`V1` 到 `V28`）都是通过主成分分析（PCA）处理过的匿名特征。只有 `Time` 和 `Amount` 是原始特征。

| 特征类型 | 数量 | 描述                               |
| -------- | ---- | ---------------------------------- |
| 匿名特征 | 28个 | `V1`, `V2`, ..., `V28` (PCA转换后) |
| 时间特征 | 1个  | `Time` (与第一笔交易的秒差)        |
| 金额特征 | 1个  | `Amount` (交易金额)                |
| 目标变量 | 1个  | `Class` (1: 欺诈, 0: 正常)         |

### 💡 关键挑战：极端不平衡

- **欺诈样本比例**：仅占总交易的 **0.172%**！
- **不平衡比例**：正常样本与欺诈样本的比例约为 **580:1**。

```python
# 欺诈交易的占比
fraud_percentage = data['Class'].value_counts(normalize=True)[1] * 100
print(f"欺诈交易占比: {fraud_percentage:.3f}%")
```
![](plot/class_distribution.png)
## ⚙️ 第二步：数据预处理

为了让模型更好地学习，我们需要对原始数据进行一些标准化处理。

1.  **特征缩放**：`Time` 和 `Amount` 特征的数值范围与其他 PCA 特征差异很大，需要进行 `StandardScaler` 标准化，使其符合标准正态分布。
2.  **数据集划分**：按照 80/20 的比例划分训练集和测试集。关键在于使用 `stratify=y` 参数，确保在训练集和测试集中，欺诈样本的比例与原始数据集保持一致，这对于不平衡学习至关重要。
```python
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

print("✅ 'Time' 和 'Amount' 特征已标准化。")

# 分离特征和目标变量
X = data.drop('Class', axis=1)
y = data['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
## 🚀 第三步：模型训练与对比实验

这是本次实战的核心。我们将构建、训练和对比多个模型，以找到最佳的欺诈检测方案。

### 模型选择

1.  **逻辑回归 (Logistic Regression)**：作为简单、快速的基线模型，用于衡量后续复杂模型的提升程度。
2.  **XGBoost (无权重)**：进行**消融实验**，观察在不处理类别不平衡的情况下，模型的表现如何。这能凸显不平衡处理的重要性。
3.  **XGBoost (有权重)**：通过设置 `scale_pos_weight` 参数来赋予以少胜多的“权重”，让模型在训练时更加关注少数的欺诈样本。
4.  **孤立森林 (Isolation Forest)**：作为一种无监督异常检测算法，它不依赖标签，而是通过识别数据点的“孤立”程度来找出异常。这在标签稀缺或不可信时尤其有用。

### ⚖️ 关键参数：`scale_pos_weight`

这个参数是 XGBoost 处理类别不平衡的利器。它的计算公式为：

\[
\text{scale-pos-weight} = \frac{\text{负样本（正常交易）数量}}{\text{正样本（欺诈交易）数量}}
\]

在我们的数据集中，这个值约为 **580**。这意味着在计算损失时，模型将一个欺诈样本的错误分类看得和 580 个正常样本的错误分类一样“严重”，从而迫使模型努力去识别每一个欺诈样本。

## 📊 第四步：模型评估与深度分析

对于欺诈检测这类问题，我们最关心的是**能否在不误伤过多正常用户的前提下，尽可能多地找出欺诈者**。因此，`AUPRC` (Area Under the Precision-Recall Curve) 是比 `ROC AUC` 更具信息量的指标。

### 性能对比总览

| Model                   | ROC AUC      | AUPRC    | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
| ----------------------- | ------------ | -------- | ----------------- | -------------- | ---------------- |
| **XGBoost With Weight** | **0.965221** | 0.875259 | 0.882979          | 0.846939       | **0.864583**     |
| XGBoost No Weight       | 0.938952     | 0.797291 | 0.866667          | 0.795918       | 0.829787         |
| Logistic Regression     | 0.957284     | 0.743949 | 0.828947          | 0.642857       | 0.724138         |
| Isolation Forest        | 0.95365      | 0.191582 | 0.304762          | 0.326531       | 0.315271         |

### 混淆矩阵
![](plot/xgboost_with_weight_confusion_matrix.png)
![](plot/xgboost_no_weight_confusion_matrix.png)
![](plot/logistic_regression_confusion_matrix.png)
![](plot/isolation_forest_confusion_matrix.png)

### 结果解读

1.  **XGBoost (有权重) 最优**：它在 `AUPRC` 和欺诈样本的 `F1-score` 上都取得了最佳表现，证明了类别加权策略的有效性。
2.  **消融实验的启示**：对比有无权重的 XGBoost，我们可以看到 `scale_pos_weight` 显著提升了召回率（从 79.6% 到 84.7%）和 F1 分数，证明了其在应对类别不平衡时的关键作用。
3.  **逻辑回归的局限**：虽然表现尚可，但其 `AUPRC` 远低于 XGBoost，说明它在精确率和召回率的权衡上能力不足。
4.  **孤立森林的挑战**：作为无监督方法，它的精确率极低（仅 3.26%），这意味着它找出的“异常”中，绝大多数都是正常交易。这凸显了在有高质量标签数据时，监督学习的巨大优势。

### Precision-Recall 曲线对比

![Precision-Recall Curve Comparison](plot/precision_recall_curve_comparison.png)


这张图直观地展示了各模型在不同阈值下的性能权衡。`XGBoost With Weight` 的曲线下面积最大，再次印证了它的优越性。

## 🔐 第五步：安全与合规提醒

- **静态代码分析**：在将类似模型部署到生产环境前，建议使用静态代码分析工具（如 [Codacy](https://www.codacy.com/)）或 GitHub Copilot Guardrails 检查代码是否存在潜在漏洞。
- **模型可解释性**：对于金融风控模型，可解释性至关重要。可以结合 SHAP (SHapley Additive exPlanations) 等工具来理解模型的决策依据。
- **免责声明**：本项目仅为技术演示，结果仅供参考。在实际金融应用中，需经过更严格的测试、验证和合规审查。

---

## 总结与思考

本次实战我们系统地解决了信用卡欺诈检测中的极端不平衡问题。通过对比实验，我们验证了 `XGBoost` 结合类别权重是处理此类问题的强大武器。

### 留给你的思考题

1.  除了 `scale_pos_weight`，还有哪些方法可以处理类别不平衡？（提示：SMOTE, ADASYN）
2.  如果完全没有标签数据，你会如何改进 `Isolation Forest` 的表现？
3.  在实际业务中，你会选择哪个模型？为什么？（提示：考虑误报的成本）

> 感谢阅读！如果你有任何问题或建议，欢迎在评论区留言。如果觉得本文对你有帮助，请不吝点赞和收藏！
