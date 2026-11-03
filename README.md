# CIFAR-10 上的轻量化 ResNet-18 与 SE 注意力实验

项目仓库：<https://github.com/ymao2554-bit/cifar10-resnet18-se-depthwise>

本项目比较四个从零训练的模型：CIFAR 适配的半宽 ResNet-18、深度可分离卷积版、SE 注意力版以及二者组合版。所有实验使用相同的数据划分、训练设置和随机种子。`results/*.json` 是报告数值的唯一来源。

## 数据

从 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 **CIFAR-10 python version**，将 `cifar-10-python.tar.gz` 放入 `data/`，解压后应存在 `data/cifar-10-batches-py/`。可用 `tar -xzf data/cifar-10-python.tar.gz -C data` 解压。原始训练集 50,000 张，按种子 42 分出 45,000 张训练和 5,000 张验证；官方测试集 10,000 张仅用于最终评估。训练采用随机裁剪、水平翻转和归一化；验证与测试只做归一化。

如官方下载较慢，也可执行 `python download_cifar.py` 从公开镜像分段下载。脚本会按官方 MD5 `c58f30108f718f92721af3b95e74349a` 校验完整文件，然后再解压。

## 运行

推荐 Python 3.12 和 NVIDIA CUDA GPU。在本目录执行：

如要使用 NVIDIA GPU，先按 [PyTorch 官网安装说明](https://pytorch.org/get-started/locally/) 安装与你的显卡驱动兼容的 CUDA 版本；随后安装其余依赖。以下命令在本次实验的 PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118 环境中完成验证。

```powershell
python -m pip install -r requirements.txt
python train.py --model baseline --epochs 30 --workers 2 --data data --out results
python train.py --model depthwise --epochs 30 --workers 2 --data data --out results
python train.py --model se --epochs 30 --workers 2 --data data --out results
python train.py --model depthwise_se --epochs 30 --workers 2 --data data --out results
python analyze.py
python predict.py path/to/image.png --model baseline
python make_report.py
python verify_report.py
```

训练脚本自动选择 CUDA 或 CPU，保存每轮损失/准确率、最优验证模型、测试预测、参数量、单张 MACs、训练时间及单线程 CPU 推理时间。`analyze.py` 根据真实日志生成学习曲线、计算量与准确率散点图、混淆矩阵及亮度/噪声扰动对比图。MACs 统计卷积与全连接的乘加次数，不包括归一化、激活和池化；CPU 延迟仅代表运行机器。

`make_report.py` 需要 `results/*.json`、`results/robustness.json` 和 `figures/*.png` 全部存在；它从这些真实结果生成 Word 初稿。在 Windows 上可运行 `finalize_wps.ps1`，由 WPS 重新保存 DOCX 并导出用于检查的 PDF。正式提交文件是 `CNN结构修改与场景适配实验报告_WPS.docx`。
`verify_report.py` 会核对 Word 中的四组测试准确率与参数量，防止报告数字偏离程序输出。

## 结果复核

报告中的数字应与 `results/*.json` 和 `results/robustness.json` 一致。可以重新运行 `analyze.py` 复核图表。网络结构见 `model.py`，训练及评估逻辑见 `train.py`。外部实拍图片仅供演示，CIFAR-10 测试准确率不能代表真实摄像头场景准确率。
