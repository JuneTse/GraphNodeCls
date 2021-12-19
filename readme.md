## 飞桨常规赛：图神经网络入门节点分类- 11月第10名方案

### 运行环境

* paddlepaddle==1.8.4
* pgl==1.2.0

### 代码内容

* model.py: 模型

  * GATGCN: GAT+GCN模型

* train.py: 训练脚本

* 训练和推理：

  ```
  python train.py
  ```

* 结果文件：submission.csv

### 模型构建思路及调优过程

* 模型结构
  * 模型结合了GAT和GCN两种结构
  * 首先，在每一层中，使用GAT和GCN分别对图的节点进行编码
  * 然后，把两种模型编码得到的节点表示相加
* 调优过程
  * 模型超参数：
    * 增加隐层维度
    * dropout设置为0
    * 训练500个epoch
* 模型效果：0.736 （注：提交时忘记保留checkpoint， 结果可能略有波动）

