## 主要文件目录
```
├─data
│  ├─interim        # 中间数据
│  │  │  colors.png
│  │  │  LULC_log.txt
│  │  │  masks.pkl
│  │  │  RSEI_log.txt
│  │  │  SVC_CNN_log.txt
│  │  │  SVC_log.txt
│  │  │  SVC_LSTM_attention_log.txt
│  │  │  SVC_LSTM_log.txt
│  │  │  white.png
│  │  │
│  │  ├─arrays
│  │  │  │  FVC.npz
│  │  │  │  LULC.npz
│  │  │  │  RSEI.npz
│  │  │  │  whole_mask.npz
│  │  │  │
│  │  │  ├─FVC
│  │  │  ├─LULC
│  │  │  └─RSEI
│  │  │
│  │  └─pics
│  │      ├─FVC
│  │      ├─LULC
│  │      └─RSEI
│  │
│  ├─processed      # 输出数据
│  │  └─output_images
│  │
│  └─raw        # 原始数据
│      │  新建 XLSX 工作表.xlsx
│      ├─FVC
│      ├─LULC
│      └─RSEI
│
├─doc
│      模型性能.docx
│      模型说明.docx
│      结果.docx
│
├─notebooks
│      CnovLSTM.ipynb       # 模型训练、推理都在notebook中处理（使用colab）
└─src
    │  analyse_res.py       # 统计结果中每个地区各级占比
    │  calculate_outputs_avg.py     # 计算结果平均值
    │  cv_utils.py      # 视觉处理的工具函数（分割地区、数据维度转换等）
    │  extract_mask.py      # 提取目标区域掩码，用于生成数据集
    │  pic_to_2d_arr.py     # 图像转换为2d数组
    │  show_color.py        # 显示色卡，调试用
    │
    ├─models
    └─visualize
           gbdt_tree_vis.py     # gdbt树结构可视化
           loss_acc_vis.py      # 模型性能可视化
           model_vis.py     # 模型可视化
```
## notebook目录结构
```
植被覆盖率预测
├─models        # 存放模型权重
│
├─outputs       # 存放输出数据
│
├─FVC.nzp       # data-interim-arrays中的二维数组
│
├─LULC.nzp       # data-interim-arrays中的二维数组
│
├─RSEI.nzp       # data-interim-arrays中的二维数组
│
└─whole_mask.nzp       # data-interim-arrays中的掩码
```
## notebook执行顺序
    1.运行mount挂载上述目录
    2.运行select model and map，选择模型和地图类型
    3.运行load data，加载选择地图类型的数据，生成数据集
    4.运行Models内的所有代码，加载所有模型类
    5.运行Train-AMP Train，加速训练模型
    6.完成训练后运行Test-Test All，生成模型性能报告、ROC、PR等
    7.运行graphs生成未来预测地图