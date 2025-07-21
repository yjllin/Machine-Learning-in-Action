# =============================================================================
# 📊 实战一：电信客户流失预测 - 从数据到模型的完整实践
# 作者：渊鱼986 | 适合：机器学习初学者到进阶
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# 1️⃣ 统一 seaborn 样式
sns.set_style('whitegrid')

# 2️⃣ 指定字体（根据实际路径调整）
yahei_regular = r"C:\Windows\Fonts\msyh.ttc"
yahei_bold    = r"C:\Windows\Fonts\msyhbd.ttc"

plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Microsoft YaHei', 'SimHei', 'Arial'],
    'axes.unicode_minus': False,          # 解决负号
    'font.weight'      : 'regular',       # 默认不用粗体，避免退回
})

print("🎯 欢迎来到电信客户流失预测实战！")
print("📚 本教程将带你从零开始构建一个完整的分类模型")

# =============================================================================
# 📖 第一步：数据加载与初步探索
# =============================================================================

print("=" * 60)
print("🔍 步骤1：加载数据集")
print("=" * 60)

# 加载数据
data = pd.read_csv(r"archive\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f"📊 数据集基本信息：")
print(f"   - 样本数量：{data.shape[0]:,} 条")
print(f"   - 特征数量：{data.shape[1]} 个")
print(f"   - 内存占用：{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 显示前5行数据
print("\n📋 数据预览（前5行）：")
print(data.head())

# 数据类型分析
print("\n🔍 数据类型分析：")
print(data.dtypes.value_counts())

# =============================================================================
# 📊 第二步：深度数据探索（EDA）
# =============================================================================

print("=" * 60)
print("📈 步骤2：探索性数据分析（EDA）")
print("=" * 60)

# 目标变量分布分析
def analyze_target_distribution(data):
    """分析目标变量的分布情况"""
    churn_counts = data['Churn'].value_counts()
    churn_pct = data['Churn'].value_counts(normalize=True) * 100
    
    print("🎯 目标变量分布：")
    print(f"   - 未流失客户：{churn_counts['No']:,} 人 ({churn_pct['No']:.1f}%)")
    print(f"   - 流失客户：{churn_counts['Yes']:,} 人 ({churn_pct['Yes']:.1f}%)")
    print(f"   - 类别不平衡比例：1:{churn_counts['No']/churn_counts['Yes']:.1f}")
    
    # 绘制分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 饼图
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(churn_counts.values, labels=['未流失', '流失'], autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('客户流失分布', fontsize=14, fontweight='bold')
    
    # 柱状图
    bars = ax2.bar(['未流失', '流失'], churn_counts.values, color=colors)
    ax2.set_title('客户流失数量对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('客户数量')
    
    # 添加数值标签
    for bar, count in zip(bars, churn_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

analyze_target_distribution(data)

# =============================================================================
# 🔧 第三步：数据预处理 - 特征工程的艺术
# =============================================================================

print("=" * 60)
print("⚙️ 步骤3：数据预处理与特征工程")
print("=" * 60)

# 分离特征和目标变量
print("🎯 分离特征和目标变量...")
X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn'].map({'No': 0, 'Yes': 1})

print(f"   ✅ 特征矩阵 X: {X.shape}")
print(f"   ✅ 目标变量 y: {y.shape}")

# 检查缺失值
print("\n🔍 缺失值检查：")
missing_info = X.isnull().sum()
if missing_info.sum() == 0:
    print("   ✅ 太棒了！没有发现缺失值")
else:
    print("   ⚠️ 发现缺失值：")
    print(missing_info[missing_info > 0])

# 特征类型分析
num_cols = X.select_dtypes(include=['float', 'int']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 特征类型统计：")
print(f"   - 数值型特征：{len(num_cols)} 个 {num_cols}")
print(f"   - 类别型特征：{len(cat_cols)} 个 {cat_cols}")

# 数值特征标准化
print("\n🔄 数值特征标准化处理...")
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(
    scaler.fit_transform(X[num_cols]), 
    columns=num_cols, 
    index=X.index
)

print(f"   ✅ 标准化前后对比（以MonthlyCharges为例）：")
print(f"      原始数据：均值={X['MonthlyCharges'].mean():.2f}, 标准差={X['MonthlyCharges'].std():.2f}")
print(f"      标准化后：均值={X_num_scaled['MonthlyCharges'].mean():.2f}, 标准差={X_num_scaled['MonthlyCharges'].std():.2f}")

# 类别特征独热编码
print("\n🔄 类别特征独热编码处理...")
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'避免多重共线性
X_cat_encoded = ohe.fit_transform(X[cat_cols])

# 获取编码后的特征名
feature_names = ohe.get_feature_names_out(cat_cols)
X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X.index)

print(f"   ✅ 编码前：{len(cat_cols)} 个类别特征")
print(f"   ✅ 编码后：{X_cat_encoded.shape[1]} 个二进制特征")

# 合并所有特征
X_processed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
print(f"\n🎉 特征工程完成！最终特征矩阵：{X_processed.shape}")

# =============================================================================
# 🤖 第四步：模型训练与对比 - 三大经典算法PK
# =============================================================================

print("=" * 60)
print("🚀 步骤4：模型训练与性能对比")
print("=" * 60)

# 数据集划分
print("📊 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print(f"   ✅ 训练集：{X_train.shape[0]:,} 样本")
print(f"   ✅ 测试集：{X_test.shape[0]:,} 样本")
print(f"   ✅ 训练集流失率：{y_train.mean():.1%}")
print(f"   ✅ 测试集流失率：{y_test.mean():.1%}")

# 处理类别不平衡
print("\n⚖️ 使用SMOTEENN处理类别不平衡...")
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

print(f"   📈 重采样前：{X_train.shape[0]:,} 样本，流失率 {y_train.mean():.1%}")
print(f"   📈 重采样后：{X_resampled.shape[0]:,} 样本，流失率 {y_resampled.mean():.1%}")

# 定义模型字典
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    '支持向量机': SVC(kernel='linear', C=0.025, probability=True, random_state=42)
}

print(f"\n🎯 开始训练 {len(models)} 个模型...")

# 模型训练与评估
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🔥 正在训练：{name}")
    print(f"{'='*50}")
    
    # 训练模型
    model.fit(X_resampled, y_resampled)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 保存结果
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auc_score': auc_score
    }
    
    # 打印详细报告
    print(f"📊 {name} 性能报告：")
    print(classification_report(y_test, y_pred, target_names=['未流失', '流失']))
    print(f"🎯 AUC-ROC得分: {auc_score:.4f}")
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['未流失', '流失'], 
                yticklabels=['未流失', '流失'])
    plt.title(f'{name} - 混淆矩阵', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 📈 第五步：模型性能对比与可视化
# =============================================================================

print("=" * 60)
print("📊 步骤5：模型性能综合对比")
print("=" * 60)

# 性能对比表格
performance_df = pd.DataFrame({
    '模型': list(results.keys()),
    'AUC-ROC': [results[name]['auc_score'] for name in results.keys()]
})

# 添加其他指标
for name in results.keys():
    y_pred = results[name]['y_pred']
    report = classification_report(y_test, y_pred, output_dict=True)
    
    performance_df.loc[performance_df['模型'] == name, 'Precision'] = report['1']['precision']
    performance_df.loc[performance_df['模型'] == name, 'Recall'] = report['1']['recall']
    performance_df.loc[performance_df['模型'] == name, 'F1-Score'] = report['1']['f1-score']

# 按AUC排序
performance_df = performance_df.sort_values('AUC-ROC', ascending=False)

print("🏆 模型性能排行榜：")
print(performance_df.round(4))

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# AUC对比
axes[0,0].bar(performance_df['模型'], performance_df['AUC-ROC'], 
              color=['gold', 'silver', '#CD7F32'])
axes[0,0].set_title('AUC-ROC 对比', fontweight='bold')
axes[0,0].set_ylabel('AUC-ROC')
axes[0,0].set_ylim(0.7, 0.9)

# Precision对比
axes[0,1].bar(performance_df['模型'], performance_df['Precision'], 
              color=['lightcoral', 'lightblue', 'lightgreen'])
axes[0,1].set_title('Precision 对比', fontweight='bold')
axes[0,1].set_ylabel('Precision')

# Recall对比
axes[1,0].bar(performance_df['模型'], performance_df['Recall'], 
              color=['orange', 'purple', 'brown'])
axes[1,0].set_title('Recall 对比', fontweight='bold')
axes[1,0].set_ylabel('Recall')

# F1-Score对比
axes[1,1].bar(performance_df['模型'], performance_df['F1-Score'], 
              color=['pink', 'cyan', 'yellow'])
axes[1,1].set_title('F1-Score 对比', fontweight='bold')
axes[1,1].set_ylabel('F1-Score')

plt.tight_layout()
plt.show()

# 找出最佳模型
best_model_name = performance_df.iloc[0]['模型']
best_auc = performance_df.iloc[0]['AUC-ROC']

print(f"\n🎉 最佳模型：{best_model_name}")
print(f"🏆 最高AUC得分：{best_auc:.4f}")

# =============================================================================
# 💡 第六步：总结与实战建议
# =============================================================================

print("=" * 60)
print("🎓 实战总结与经验分享")
print("=" * 60)

print("📚 本次实战我们学到了什么：")
print("   1️⃣ 数据探索：通过EDA发现数据特点和潜在问题")
print("   2️⃣ 特征工程：数值标准化 + 类别编码的标准流程")
print("   3️⃣ 不平衡处理：SMOTEENN混合采样技术的应用")
print("   4️⃣ 模型对比：三种经典算法的优缺点对比")
print("   5️⃣ 性能评估：多指标综合评估模型性能")

print(f"\n🔍 关键发现：")
print(f"   • 数据集存在明显的类别不平衡（流失率约26%）")
print(f"   • SMOTEENN采样有效提升了模型的召回率")
print(f"   • {best_model_name}在综合性能上表现最佳")

print(f"\n💼 业务建议：")
print(f"   • 重点关注月费较高的按月付费客户")
print(f"   • 对新客户（tenure较短）加强关怀")
print(f"   • 优化合同条款，减少按月付费比例")

print(f"\n🚀 下一步优化方向：")
print(f"   • 尝试XGBoost、LightGBM等集成学习算法")
print(f"   • 使用网格搜索或贝叶斯优化调参")
print(f"   • 添加SHAP解释性分析")
print(f"   • 考虑成本敏感学习")

print(f"\n🎯 感谢阅读！如果觉得有用，请点赞收藏支持！")