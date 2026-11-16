# 比较概率模型与实体分组

运行环境为 Python 3.11。材料使用合成比较向量，不需要下载模型或数据。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py model
python demo.py cluster
python demo.py link
```

`comparison_model.py` 当前提供 `IndependentComparisonModel.fit(rows, dimensions, alpha)`、`likelihood(values, label)` 和 `score(values, prior)`。`clustering.py` 当前提供 `mutual_best_partition(nodes, weights, duplicate_free_sources)`；`weights`是以两个节点ID的tuple为键的dict。二者可直接导入，`demo.py`演示分别调用和组合调用。

`fixtures/training.json` 保存各组带标签比较样本；每个case都有dimensions、beta、rows。`fixtures/clustering.json` 保存不需要模型的节点、来源约束和边权，JSON中的一条edge为[left, right, weight]。`fixtures/linkage.json`保存普通联动，以及外部联合概率表、对应pair列表和一个修改后的pair。向量顺序按二进制字典序，None在JSON中写作null。

可以分别运行现有材料中的场景：

```bash
python demo.py model --case partial
python demo.py model --case all_missing
python demo.py cluster --case two_sources
python demo.py cluster --case three_sources_signed
```

请按任务正文扩展模型与聚类行为，并在本文件补充修改后的独立拟合、外部表评分、外部边权聚类及联动调用方法；类名或布局可以调整。最终演示应使用实际被聚类消费的边权，并记录目标值与分组。说明所选估计方法、停止设置、分组算法和复现命令。
