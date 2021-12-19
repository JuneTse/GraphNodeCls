#!/usr/bin/env python
# coding: utf-8

# ## 代码整体逻辑
# 
# 1. 读取提供的数据集，包含构图以及读取节点特征（用户可自己改动边的构造方式）
# 
# 2. 配置化生成模型，用户也可以根据教程进行图神经网络的实现。
# 
# 3. 开始训练
# 
# 4. 执行预测并产生结果文件
# 

# ## 环境配置
# 
# 该项目依赖飞桨paddlepaddle==1.8.4, 以及pgl==1.2.0。请按照版本号下载对应版本就可运行。


import pgl
import paddle.fluid as fluid
import numpy as np
import time
import pandas as pd



from easydict import EasyDict as edict

config = {
    "model_name": "GATGCN",
    "num_layers": 2,
    "dropout": 0.0,
    "hidden_size":64,
    "learning_rate": 0.005,
    "weight_decay": 1e-8,
    "edge_dropout": 0.00,
}

# config = {
#     "model_name": "GCN",
#     "num_layers": 2,
#     "dropout": 0.2,
#     "hidden_size":512,
#     "learning_rate": 0.001,
#     "weight_decay": 0.00001,
#     "edge_dropout": 0.00,
# }

config = edict(config)


# ## 数据加载模块
# 
# 这里主要是用于读取数据集，包括读取图数据构图，以及训练集的划分。

# In[16]:


from collections import namedtuple

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):
    # 从数据中读取边
    edges = pd.read_csv("work/edges.csv", header=None, names=["src", "dst"]).values

    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges

def load():
    # 从数据中读取点特征和边，以及数据划分
    node_feat = np.load("work/feat.npy")
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)

    
    df = pd.read_csv("work/train.csv")
    node_index = df["nid"].values
    node_label = df["label"].values
    train_part = int(len(node_index) * 0.95)
    train_index = node_index[:train_part]
    train_label = node_label[:train_part]
    valid_index = node_index[train_part:]
    valid_label = node_label[train_part:]
    test_index = pd.read_csv("work/test.csv")["nid"].values


    feat_label=np.zeros(shape=[num_nodes,35])
    feat_label[node_index,node_label]=1


    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})
    indegree = graph.indegree()
    norm = np.maximum(indegree.astype("float32"), 1)
    norm = np.power(norm, -0.5)
    graph.node_feat["norm"] = np.expand_dims(norm, -1)

    dataset = Dataset(graph=graph, 
                    train_label=train_label,
                    train_index=train_index,
                    valid_index=valid_index,
                    valid_label=valid_label,
                    test_index=test_index, num_classes=35)
    return dataset


# In[17]:


dataset = load()

train_index = dataset.train_index
train_label = np.reshape(dataset.train_label, [-1 , 1])
train_index = np.expand_dims(train_index, -1)

val_index = dataset.valid_index
val_label = np.reshape(dataset.valid_label, [-1, 1])
val_index = np.expand_dims(val_index, -1)

test_index = dataset.test_index
test_index = np.expand_dims(test_index, -1)
test_label = np.zeros((len(test_index), 1), dtype="int64")


# ## 组网模块
# 
# 这里是组网模块，目前已经提供了一些预定义的模型，包括**GCN**, **GAT**, **APPNP**等。可以通过简单的配置，设定模型的层数，hidden_size等。你也可以深入到model.py里面，去奇思妙想，写自己的图神经网络。

# In[18]:


import pgl
import model
import paddle.fluid as fluid
import numpy as np
import time
from build_model import build_model

# 使用CPU
#place = fluid.CPUPlace()

# 使用GPU
place = fluid.CUDAPlace(0)

train_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
with fluid.program_guard(train_program, startup_program):
    with fluid.unique_name.guard():
        gw, loss, acc, pred = build_model(dataset,
                            config=config,
                            phase="train",
                            main_prog=train_program)

test_program = fluid.Program()
with fluid.program_guard(test_program, startup_program):
    with fluid.unique_name.guard():
        _gw, v_loss, v_acc, v_pred = build_model(dataset,
            config=config,
            phase="test",
            main_prog=test_program)


test_program = test_program.clone(for_test=True)

exe = fluid.Executor(place)


# ## 开始训练过程
# 
# 图神经网络采用FullBatch的训练方式，每一步训练就会把所有整张图训练样本全部训练一遍。
# 

epoch = 500
exe.run(startup_program)

# 将图数据变成 feed_dict 用于传入Paddle Excecutor
feed_dict = gw.to_feed(dataset.graph)
best_acc=0
for epoch in range(epoch):
    # Full Batch 训练
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(train_index, dtype="int64")
    feed_dict["node_label"] = np.array(train_label, dtype="int64")
    
    train_loss, train_acc = exe.run(train_program,
                                feed=feed_dict,
                                fetch_list=[loss, acc],
                                return_numpy=True)

    # Full Batch 验证
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(val_index, dtype="int64")
    feed_dict["node_label"] = np.array(val_label, dtype="int64")
    val_loss, val_acc = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_loss, v_acc],
                            return_numpy=True)
    print("Epoch", epoch, "Train Acc", train_acc[0], "Valid Acc", val_acc[0])
    if val_acc[0]>best_acc:
        best_acc=val_acc[0]
        if epoch>100:
            # ## 对测试集进行预测
            # 
            # 训练完成后，我们对测试集进行预测。预测的时候，由于不知道测试集合的标签，我们随意给一些测试label。最终我们获得测试数据的预测结果。

            feed_dict["node_index"] = np.array(test_index, dtype="int64")
            feed_dict["node_label"] = np.array(test_label, dtype="int64") #假标签
            test_prediction = exe.run(test_program,
                                        feed=feed_dict,
                                        fetch_list=[v_pred],
                                        return_numpy=True)[0]

            # ## 生成提交文件
            # 
            # 最后一步，我们可以使用pandas轻松生成提交文件，最后下载 submission.csv 提交就好了。

            submission = pd.DataFrame(data={
                                        "nid": test_index.reshape(-1),
                                        "label": test_prediction.reshape(-1)
                                    })
            submission.to_csv("submission.csv", index=False)




