这里主要是IN你的网络架构与我自己的实验部分代码

程序默认在 python 3.8上运行

## 文件目录说明

```
├── ckpt/               # 存放训练后的模型
├── data/
│   ├── generate_Glucose_sim.py      # 实验的环境（血糖模拟器） 看主程序就行 
│   ├── vpatient_params.csv     # 存放不同人的不同属性数据
│   └── Glucose_sim_data004_(0-9).npy    # 基于adults004环境的不同胰岛素分布的九组数据
├── baseline/            # 包含baseline涉及到的不同网络架构
├── casual_tree/         # 树和图模型
│   ├── Tree.py      # 定义树结构
│   └── Graph.py     # 由因果图通过Floyd算法转树，从而确定网络输入的分层信息
├── Net        # INN相关网络架构
│   ├── couple_layers.py     # 流模型里的Real_NVP
│   ├── OWN.py               # 正交可逆阵计算方法
│   └── OWN_Linear           # 正交层
├── G2M_model.py        # INN网络架构与forward计算
└── main.py          # 我的实验部分主程序
 ```

以上其他文件除utils以外尚未提及的是没啥用的