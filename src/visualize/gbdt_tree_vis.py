from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.datasets import make_classification

# 1. 创建示例数据集
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# 2. 训练GBDT模型
gbdt = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
gbdt.fit(X, y)

# 3. 导出第一棵决策树
tree = gbdt.estimators_[0, 0]  # 获取第0棵树

# 4. 使用 graphviz 可视化
dot_data = export_graphviz(tree, out_file=None, feature_names=['feature1', 'feature2', 'feature3', 'feature4'], 
                           filled=True, rounded=True, special_characters=True)

# 渲染图形
graph = graphviz.Source(dot_data)
graph.render("gbdt_tree")  # 保存为文件
graph.view()  # 打开图形