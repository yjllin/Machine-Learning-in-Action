# =============================================================================
# 📊 实战二：信用卡欺诈检测 - 应对极端不平衡数据
# 作者：渊鱼986 | 适合：机器学习进阶者
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve
import warnings
import os
warnings.filterwarnings('ignore')

# 1️⃣ 统一 seaborn 样式
sns.set_style('whitegrid')

# 2️⃣ 指定字体（根据实际路径调整）
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

print("🎯 欢迎来到信用卡欺诈检测实战！")
print("🔥 本项目将挑战极端不平衡数据下的分类问题。")

# =============================================================================
# 📖 第一步：数据加载与初步探索
# =============================================================================

print("=" * 60)
print("🔍 步骤1：加载数据集")
print("=" * 60)

# 加载数据
try:
    data = pd.read_csv(r"data/creditcard.csv")
except FileNotFoundError:
    print("错误：找不到 data/creditcard.csv 文件。")
    print("请先从 Kaggle 下载数据集并放到 'data' 目录下：")
    print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    exit()

print(f"📊 数据集基本信息：")
print(f"   - 样本数量：{data.shape[0]:,} 条")
print(f"   - 特征数量：{data.shape[1]} 个")

# 目标变量分布分析
fraud_counts = data['Class'].value_counts()
fraud_pct = data['Class'].value_counts(normalize=True) * 100

print("\n🎯 目标变量分布：")
print(f"   - 正常交易：{fraud_counts[0]:,} 笔 ({fraud_pct[0]:.3f}%)")
print(f"   - 欺诈交易：{fraud_counts[1]:,} 笔 ({fraud_pct[1]:.3f}%)")
print(f"   - 类别不平衡比例：约 1:{fraud_counts[0]/fraud_counts[1]:.0f}")

# 创建 plot 目录用于存放图表
if not os.path.exists('plot'):
    os.makedirs('plot')

# =============================================================================
# 🔧 第二步：数据预处理
# =============================================================================

print("=" * 60)
print("⚙️ 步骤2：数据预处理")
print("=" * 60)

# 标准化 'Amount' 和 'Time' 特征
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

print(f"\n📊 数据集划分完毕：")
print(f"   - 训练集: {X_train.shape[0]:,} 样本")
print(f"   - 测试集: {X_test.shape[0]:,} 样本")


def visualize_class_distribution(data):
    """可视化目标变量'Class'的分布情况"""
    class_counts = data['Class'].value_counts()
    
    print("\n📊 正在生成目标变量分布图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 甜甜圈图
    colors = ['#66b3ff', '#ff9999']
    labels = ['正常 (Class 0)', '欺诈 (Class 1)']
    ax1.pie(class_counts, labels=labels, autopct='%1.3f%%', 
            colors=colors, startangle=140, wedgeprops=dict(width=0.3), pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax1.set_title('交易类别分布 (甜甜圈图)', fontsize=16, fontweight='bold')
    
    # 柱状图 (使用对数刻度)
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax2, palette=colors)
    ax2.set_title('交易类别数量对比', fontsize=16, fontweight='bold')
    ax2.set_xlabel('交易类别', fontsize=12)
    ax2.set_ylabel('交易数量 (对数刻度)', fontsize=12)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels)
    ax2.set_yscale('log')
    
    # 添加数值标签
    for i, count in enumerate(class_counts.values):
        ax2.text(i, count, f'{count:,}', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    save_path = 'plot/class_distribution.png'
    plt.savefig(save_path)
    print(f"🖼️  目标变量分布图已保存至 '{save_path}'")
    plt.close() # 关闭图像，避免在脚本运行时显示

# 调用可视化函数
visualize_class_distribution(data)

# =============================================================================
# 🤖 第三步：模型训练与对比
# =============================================================================

print("=" * 60)
print("🚀 步骤3：模型训练与对比")
print("=" * 60)

# 定义一个字典来存储所有模型的结果
results = {}

# --- 模型1：逻辑回归 (基线模型) ---
print("\n⏳ 1. 训练逻辑回归模型...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("✅ 逻辑回归模型训练完成。")

# --- 模型2：XGBoost (无类别权重 - 消融实验) ---
print("\n⏳ 2. 训练 XGBoost (无权重)... ")
xgb_no_weight = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_no_weight.fit(X_train, y_train)
print("✅ XGBoost (无权重) 模型训练完成。")

# --- 模型3：XGBoost (有类别权重) ---
print("\n⏳ 3. 训练 XGBoost (有权重)... ")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"⚖️ 类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")
xgb_model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
print("✅ XGBoost (有权重) 模型训练完成。")

# =============================================================================
# 📈 第四步：模型评估与对比
# =============================================================================

print("=" * 60)
print("📊 步骤4：模型性能评估与对比")
print("=" * 60)

# 评估函数
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    # 对于逻辑回归和XGBoost，我们关心概率
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probas)
        auprc = average_precision_score(y_test, probas)
    else: # 对于Isolation Forest
        # decision_function返回异常分数，分数越低越异常
        scores = model.decision_function(X_test)
        # 将分数转换为类似概率的度量（越小越可能是欺诈）
        probas = -scores
        roc_auc = roc_auc_score(y_test, probas)
        auprc = average_precision_score(y_test, probas)

    report = classification_report(y_test, predictions, target_names=['正常', '欺诈'], output_dict=True)
    
    results[model_name] = {
        'ROC AUC': roc_auc,
        'AUPRC': auprc,
        'Precision (Fraud)': report['欺诈']['precision'],
        'Recall (Fraud)': report['欺诈']['recall'],
        'F1-score (Fraud)': report['欺诈']['f1-score']
    }
    
    print(f"\n--- 评估: {model_name} ---")
    print(classification_report(y_test, predictions, target_names=['正常', '欺诈']))
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '欺诈'], yticklabels=['正常', '欺诈'])
    plt.title(f'{model_name} 混淆矩阵', fontsize=16)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    file_name = f"plot/{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
    plt.savefig(file_name)
    print(f"🖼️  混淆矩阵已保存为 '{file_name}'")
    plt.close()
    return probas

# 评估所有监督模型
probas_lr = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
probas_xgb_no_weight = evaluate_model(xgb_no_weight, X_test, y_test, 'XGBoost No Weight')
probas_xgb = evaluate_model(xgb_model, X_test, y_test, 'XGBoost With Weight')

# 绘制 Precision-Recall 曲线对比
plt.figure(figsize=(10, 8))

precision, recall, _ = precision_recall_curve(y_test, probas_lr)
plt.plot(recall, precision, marker='.', label='Logistic Regression')

precision, recall, _ = precision_recall_curve(y_test, probas_xgb_no_weight)
plt.plot(recall, precision, marker='.', label='XGBoost No Weight')

precision, recall, _ = precision_recall_curve(y_test, probas_xgb)
plt.plot(recall, precision, marker='.', label='XGBoost With Weight')

plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('模型 Precision-Recall 曲线对比')
plt.legend()
plt.grid(True)
plt.savefig('plot/precision_recall_curve_comparison.png')
print("\n🖼️  Precision-Recall 曲线对比图已保存为 'plot/precision_recall_curve_comparison.png'")
plt.close()

# =============================================================================
# 🌲 第五步：模型训练 - Isolation Forest (无监督)
# =============================================================================

print("=" * 60)
print("🌲 步骤5：使用 Isolation Forest 进行无监督异常检测")
print("=" * 60)

# contamination 参数设置为数据集中已知的欺诈率
contamination_rate = float(fraud_pct[1] / 100)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1
)

print("\n⏳ 开始训练 Isolation Forest 模型...")
iso_forest.fit(X_train)
print("✅ Isolation Forest 模型训练完成。")

# 评估 Isolation Forest
iso_preds_binary = [1 if p == -1 else 0 for p in iso_forest.predict(X_test)]

# 对于无监督模型，我们单独评估并添加到结果中
report_iso = classification_report(y_test, iso_preds_binary, target_names=['正常', '欺诈'], output_dict=True)
scores_iso = -iso_forest.decision_function(X_test)
roc_auc_iso = roc_auc_score(y_test, scores_iso)
auprc_iso = average_precision_score(y_test, scores_iso)

results['Isolation Forest'] = {
    'ROC AUC': roc_auc_iso,
    'AUPRC': auprc_iso,
    'Precision (Fraud)': report_iso['欺诈']['precision'],
    'Recall (Fraud)': report_iso['欺诈']['recall'],
    'F1-score (Fraud)': report_iso['欺诈']['f1-score']
}

print("\n--- 评估: Isolation Forest ---")
print(classification_report(y_test, iso_preds_binary, target_names=['正常', '欺诈']))
print(f"ROC AUC: {roc_auc_iso:.4f}")
print(f"AUPRC: {auprc_iso:.4f}")

cm_iso = confusion_matrix(y_test, iso_preds_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['正常', '欺诈'], yticklabels=['正常', '欺诈'])
plt.title('Isolation Forest 混淆矩阵', fontsize=16)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.savefig('plot/isolation_forest_confusion_matrix.png')
print("🖼️  Isolation Forest 混淆矩阵已保存为 'plot/isolation_forest_confusion_matrix.png'")
plt.close()

# =============================================================================
# 🏆 第六步：结果汇总与对比
# =============================================================================

print("=" * 60)
print("🏆 步骤6：所有模型性能汇总")
print("=" * 60)

results_df = pd.DataFrame(results).T.sort_values(by='AUPRC', ascending=False)

print(results_df.to_markdown())

print("\n🎉 实战完成！请查看生成的图片、评估报告和最终的性能对比表。")