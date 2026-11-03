# CIFAR-10 上的轻量化 ResNet-18 与 SE 注意力实验

项目仓库：<https://github.com/ymao2554-bit/cifar10-resnet18-se-depthwise>

本项目比较四个从零训练的模型：CIFAR 适配的半宽 ResNet-18、深度可分离卷积版、SE 注意力版以及二者组合版。所有实验使用相同的数据划分、训练设置和随机种子。`results/*.json` 是报告数值的唯一来源。

仓库保留代码、模型权重、实验结果和图表。课程报告作为本地交付文件保存，仓库不包含成品 Word 文档及其生成脚本。下方命令用于复现模型实验。

## 数据

从 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 **CIFAR-10 python version**，将 `cifar-10-python.tar.gz` 放入 `data/`，解压后应存在 `data/cifar-10-batches-py/`。可用 `tar -xzf data/cifar-10-python.tar.gz -C data` 解压。原始训练集 50,000 张，按种子 42 分出 45,000 张训练和 5,000 张验证，官方测试集 10,000 张仅用于最终评估。训练采用随机裁剪、水平翻转和归一化，验证与测试只做归一化。

如官方下载较慢，也可执行 `python download_cifar.py` 从公开镜像分段下载（我的做法。。。真没招了。。。）。脚本会按官方 MD5 `c58f30108f718f92721af3b95e74349a` 校验完整文件，然后再解压。

## 运行

推荐 Python 3.12 和 NVIDIA CUDA GPU。在本目录执行：

如要使用 NVIDIA GPU，先按 [PyTorch 官网安装说明](https://pytorch.org/get-started/locally/) 安装与你的显卡驱动兼容的 CUDA 版本，随后安装其余依赖。以下命令在本次实验的 PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118 环境中完成验证。

```powershell
python -m pip install -r requirements.txt
python train.py --model baseline --epochs 30 --workers 2 --data data --out results
python train.py --model depthwise --epochs 30 --workers 2 --data data --out results
python train.py --model se --epochs 30 --workers 2 --data data --out results
python train.py --model depthwise_se --epochs 30 --workers 2 --data data --out results
python analyze.py
python predict.py path/to/image.png --model baseline
```

训练脚本自动选择 CUDA 或 CPU，保存每轮损失/准确率、最优验证模型、测试预测、参数量、单张 MACs、训练时间及单线程 CPU 推理时间。`analyze.py` 根据真实日志生成学习曲线、计算量与准确率散点图、混淆矩阵及亮度/噪声扰动对比图。MACs 统计卷积与全连接的乘加次数，不包括归一化、激活和池化。CPU 延迟仅代表运行机器。

