"""Build the coursework DOCX solely from verified experiment output JSON."""
import json
from pathlib import Path
import torch

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).parent
NAMES = ["baseline", "depthwise", "se", "depthwise_se"]
CN = {"baseline":"基线 ResNet-18", "depthwise":"深度可分离卷积", "se":"SE 注意力", "depthwise_se":"深度可分离卷积加 SE"}
results = {n:json.loads((ROOT/"results"/f"{n}.json").read_text(encoding="utf-8")) for n in NAMES}
robust = json.loads((ROOT/"results"/"robustness.json").read_text(encoding="utf-8"))
repo_file = ROOT / 'repository_url.txt'
repo_url = repo_file.read_text(encoding='utf-8').strip() if repo_file.exists() else '尚未发布'
assert all(results[n]["test"]["total"] == 10000 for n in NAMES)
assert len({results[n]["seed"] for n in NAMES}) == 1
assert len({results[n]["epochs"] for n in NAMES}) == 1

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = Inches(.73); sec.bottom_margin = Inches(.70)
sec.left_margin = Inches(.82); sec.right_margin = Inches(.75)

def font(style, family, size, bold=False, color=(20,30,45)):
    style.font.name = family
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(*color)
    style._element.rPr.rFonts.set(qn('w:eastAsia'), family)

normal=doc.styles['Normal']; font(normal,'宋体',12)
normal.paragraph_format.line_spacing=1.5
normal.paragraph_format.first_line_indent=Pt(24)
normal.paragraph_format.space_after=Pt(0)
for key,size,before,after in [('Title',16,16,16),('Heading 1',16,16,16),('Heading 2',14,7,7)]:
    st=doc.styles[key]; font(st,'黑体',size,True,(0,0,0))
    st.paragraph_format.space_before=Pt(before); st.paragraph_format.space_after=Pt(after)
    st.paragraph_format.keep_with_next=True
    st.paragraph_format.first_line_indent=Pt(0)
    ppr = st._element.get_or_add_pPr()
    for border in list(ppr):
        if border.tag == qn('w:pBdr'):
            ppr.remove(border)

def para(text='', bold_prefix=None):
    p=doc.add_paragraph(style='Normal')
    if bold_prefix and text.startswith(bold_prefix):
        p.add_run(bold_prefix).bold=True; p.add_run(text[len(bold_prefix):])
    else:p.add_run(text)
    return p

def h1(text): doc.add_heading(text,1)
def h2(text): doc.add_heading(text,2)
def page(title):
    doc.add_page_break(); h1(title)

def caption(text):
    p=doc.add_paragraph()
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent=Pt(0)
    p.paragraph_format.space_before=Pt(5); p.paragraph_format.space_after=Pt(8)
    r=p.add_run(text); r.font.name='宋体'; r.font.size=Pt(10.5)
    r._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')

def figure(filename, title, width=6.4):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent=Pt(0)
    p.add_run().add_picture(str(ROOT/'figures'/filename), width=Inches(width))
    caption(title)

def table(headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.style='Table Grid'
    t.autofit=False
    if widths:
        for c,w in zip(t.rows[0].cells,widths): c.width=Inches(w)
    for i,x in enumerate(headers):t.rows[0].cells[i].text=str(x)
    for row in rows:
        cells=t.add_row().cells
        for i,x in enumerate(row): cells[i].text=str(x)
    for ri,row in enumerate(t.rows):
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.15
                p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
                for run in p.runs:
                    run.font.name='宋体';run.font.size=Pt(9.5)
                    run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
                    if ri==0:run.bold=True;run.font.color.rgb=RGBColor(255,255,255)
            tcPr=cell._tc.get_or_add_tcPr()
            shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'1F4E79' if ri==0 else ('F0F5FA' if ri%2==0 else 'FFFFFF'))
            tcPr.append(shd)
    return t

def pct(x):return f'{100*x:.2f}%'
def mi(x):return f'{x/1e6:.3f}'
base=results['baseline']; best=max(results.values(),key=lambda r:r['test']['accuracy'])

title=doc.add_paragraph(style='Title');title.alignment=WD_ALIGN_PARAGRAPH.CENTER
title.add_run('CIFAR-10 图像分类中的轻量化 ResNet-18 与 SE 注意力实验')
for border in list(title._p.get_or_add_pPr()):
    if border.tag == qn('w:pBdr'):
        title._p.get_or_add_pPr().remove(border)
caption('机器学习大作业实验报告')
para('研究对象：CIFAR-10 十类图像分类；基础架构：面向 32×32 输入的半宽 ResNet-18。')
para('实验目的：在统一训练设置下，比较标准卷积、深度可分离卷积与 SE 通道注意力对分类精度、模型规模、计算量和 CPU 推理时间的影响。')
para(f'实验环境：PyTorch {torch.__version__}，NVIDIA GeForce RTX 4060 Laptop GPU；随机种子 {base["seed"]}；每组训练 {base["epochs"]} 轮。')
h2('摘要')
para(f'以官方测试集 10,000 张图片为最终评估对象，基线模型正确识别 {base["test"]["correct"]} 张，Top-1 准确率为 {pct(base["test"]["accuracy"])}。四组模型中准确率最高者为{CN[best["name"]]}，达到 {pct(best["test"]["accuracy"])}。报告中的统计数值由保存的训练结果 JSON 直接生成。')
para('关键词：卷积神经网络；ResNet-18；深度可分离卷积；SE 注意力；CIFAR-10')

page('一 任务与实验路线')
para('本作业实现一个经典 CNN 图像分类架构，完成基础训练、结构修改、对照实验与场景应用。选择残差网络是因为它保留卷积层的核心计算，同时可以在每个残差块内进行受控替换。项目不采用 UNet 或其嵌套变体。')
h2('实验问题')
para('第一，半宽 ResNet-18 在 CIFAR-10 上能达到怎样的基线准确率。第二，将 3×3 标准卷积替换为深度可分离卷积后，参数量与乘加次数的减少是否伴随精度损失。第三，加入 SE 通道注意力能否改善识别。第四，组合结构能否取得更合适的精度与计算成本。')
h2('公平比较原则')
para(f'四组模型采用相同的训练集、验证集、测试集、随机种子 {base["seed"]}、批量大小 {base["batch"]}、优化器与训练轮数。训练过程中根据验证集准确率保存检查点，最后只对保存的最优检查点进行测试集评估。单次随机种子的结果只用于本次实验比较，不解释为统计显著差异。')
table(['模型','标准卷积','深度可分离卷积','SE 模块'],[['基线','是','否','否'],['轻量化','否','是','否'],['注意力','是','否','是'],['组合','否','是','是']])
caption('表 1 四组受控实验的结构差异')

page('二 数据集与数据划分')
para('CIFAR-10 是公开的十类彩色图像数据集。每张图像为 32×32 像素，共 60,000 张；官方训练集 50,000 张，官方测试集 10,000 张。类别包括飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。数据来自多伦多大学发布的 Python 版本，并在本项目中保留原始归档。')
h2('训练 验证 测试')
para('从官方训练集中用固定随机种子抽取 5,000 张作验证集，剩余 45,000 张作训练集。官方测试集始终独立，不参与调参和检查点选择。训练集采用边缘填充后的随机裁剪与随机水平翻转；全部输入再按 CIFAR-10 通道均值和标准差归一化。验证和测试阶段不采用随机增强。')
table(['划分','图片数','用途'],[['训练集','45,000','反向传播更新参数'],['验证集','5,000','选择最优检查点'],['测试集','10,000','最终一次性报告指标']])
caption('表 2 数据划分与用途')
figure('data_distribution.png','图 1 各类别在训练 验证 测试集中的实际样本数',6.1)
para('数据集下载包只需 cifar-10-python.tar.gz。项目读取其解压后的 cifar-10-batches-py 文件夹。运行说明给出了获取、解压和完整复现的命令。')

page('三 基础 CNN 架构')
para('基础模型采用 ResNet-18 的 2、2、2、2 个基本残差块分布。考虑到 CIFAR-10 图像仅 32×32，输入端使用 3×3、步长 1 的卷积，不使用面向大尺寸图像的 7×7 卷积和初始最大池化。四个阶段的通道宽度为 32、64、128、256，因此本实验的基线是半宽 CIFAR 适配版 ResNet-18。')
table(['组件','配置','作用'],[['输入卷积','3×3，3→32','提取局部纹理'],['残差阶段','每阶段 2 块，通道 32/64/128/256','逐层形成高级特征'],['捷径分支','通道或尺寸变化时采用 1×1 卷积','保证残差相加维度一致'],['分类头','全局平均池化 + 全连接','输出十类得分']])
caption('表 3 基础网络的实际配置')
figure('architecture.png','图 2 实际实现的半宽 ResNet-18 数据流',6.2)
para(f'代码统计的基线参数量为 {base["parameters"]:,}，单张 32×32 图像卷积及全连接层乘加量为 {base["macs"]:,} MACs。MACs 不包括归一化、激活函数和池化，因此它是主要算子的计算代理指标，而非实际耗时。')

page('四 轻量化结构修改')
para('深度可分离卷积把一次 3×3 标准卷积拆为逐通道 3×3 卷积和逐点 1×1 卷积。代码使用 groups 等于输入通道数实现逐通道卷积，然后用 1×1 卷积完成通道混合。残差捷径、归一化和分类头保持同样规则。')
h2('实际规模变化')
dw=results['depthwise']
table(['指标','基线','深度可分离卷积','变化'],[['参数量',f'{base["parameters"]:,}',f'{dw["parameters"]:,}',f'{100*(dw["parameters"]/base["parameters"]-1):.2f}%'],['MACs',f'{base["macs"]:,}',f'{dw["macs"]:,}',f'{100*(dw["macs"]/base["macs"]-1):.2f}%'],['测试准确率',pct(base['test']['accuracy']),pct(dw['test']['accuracy']),f'{100*(dw["test"]["accuracy"]-base["test"]["accuracy"]):+.2f} 个百分点']])
caption('表 4 轻量化结构与基线的实际比较')
figure('compression.png','图 3 四组结构的参数量与单张计算量',6.1)
para('参数量和 MACs 的下降由结构本身决定；准确率变化则依据真实测试结果。CPU 推理时间另外实测，不能仅凭 MACs 推断运行一定更快，因为不同算子可能具有不同的内存访问和框架开销。')

page('五 SE 注意力及组合结构')
para('SE 模块放置在残差块第二个卷积和归一化之后、与捷径相加之前。它先对每个通道做全局平均池化，再经过两个 1×1 卷积和 Sigmoid 得到通道权重，将原特征按通道缩放。缩减率为 16，隐藏通道数最少为 4。')
se=results['se']; combo=results['depthwise_se']
table(['模型','参数量','MACs','测试准确率'],[[CN[n],f'{results[n]["parameters"]:,}',f'{results[n]["macs"]:,}',pct(results[n]['test']['accuracy'])] for n in NAMES])
caption('表 5 四组模型的结构规模与最终准确率')
figure('se_effect.png','图 4 加入 SE 前后的准确率与 CPU 延迟',6.0)
para(f'单独加入 SE 后，参数量从 {base["parameters"]:,} 变为 {se["parameters"]:,}，测试准确率变化为 {100*(se["test"]["accuracy"]-base["test"]["accuracy"]):+.2f} 个百分点。组合结构的参数量为 {combo["parameters"]:,}，用于判断轻量化以后通道重标定是否还有收益。')
para('这里的创新是对基础 CNN 进行结构组合和受控实验，不把深度可分离卷积或 SE 模块表述为本作业首次提出的方法。')

page('六 训练设置与调优过程')
para(f'四个实验从随机初始化开始训练 {base["epochs"]} 轮，批量大小 {base["batch"]}。优化器为带 0.9 动量的 SGD，初始学习率 0.1，权重衰减 5×10⁻⁴；学习率按余弦退火降低。GPU 训练启用自动混合精度。模型检查点由验证集准确率最高的一轮确定。')
table(['设置','取值或方法'],[['设备','NVIDIA GeForce RTX 4060 Laptop GPU'],['输入大小','3×32×32'],['随机种子',str(base['seed'])],['数据加载进程','2'],['损失函数','交叉熵'],['数据增强','随机裁剪、水平翻转'],['选择准则','验证集 Top-1 准确率最高']])
caption('表 6 可复现的统一训练设置')
figure('loss_curves.png','图 5 四组模型的验证损失曲线',6.0)
para('调优重点放在输入端与任务尺度匹配、训练学习率变化以及验证集模型选择。为了保持结构消融的可解释性，各模型不单独调整超参数。程序逐轮保存训练损失、训练准确率、验证损失和验证准确率，供报告中的曲线与结果检查。')

page('七 收敛过程与基线结果')
figure('learning_curves.png','图 6 四组模型的验证集准确率曲线',6.3)
para(f'基线最佳验证准确率为 {pct(base["best_val_accuracy"])}，对应检查点在官方测试集达到 {pct(base["test"]["accuracy"])}。曲线反映在统一训练预算下各结构的学习过程。若个别曲线波动，不能单凭单轮峰值判断结构优劣，应结合最终测试结果和成本共同分析。')

page('八 精度与计算成本对比')
figure('tradeoff.png','图 7 单张计算量与测试准确率的关系',6.3)
table(['模型','参数 万','MACs 百万','CPU 毫秒/张'],[[CN[n],f'{results[n]["parameters"]/1e4:.2f}',f'{results[n]["macs"]/1e6:.2f}',f'{results[n]["cpu_latency_ms"]:.3f}'] for n in NAMES])
caption('表 7 模型规模与单线程 CPU 推理时间')
para('CPU 延迟采用同一设备、单线程、预热后重复推理的均值，反映本机条件。若轻量模型 MACs 明显降低而延迟改善有限，原因可能是逐通道卷积的算子开销和内存访问；该解释是依据数据的推断，不等于独立的硬件剖析结论。')

page('九 类别错误分析')
figure('confusion.png',f'图 8 准确率最高模型 {CN[best["name"]]} 的测试集混淆矩阵',5.8)
para('矩阵的行表示真实类别，列表示预测类别。对角线是正确识别数，非对角线表示混淆。该图直接使用保存的 10,000 个测试标签与预测结果生成，没有从训练集抽样代替测试集。具体易混类别可由图中的非对角线高值定位，再结合代表性样本判断视觉特征是否相似。')

page('十 场景适配与稳健性')
para('应用场景设定为离线终端上的十类小图像分类。训练在 GPU 上进行，部署评估则关注 CPU 的模型大小和单张推理时间。为近似拍摄条件变化，给官方测试集加入两种固定扰动：图像亮度乘以 0.65，以及像素空间标准差为 0.05 的高斯噪声。扰动只用于评估，不参与训练。')
figure('robustness.png','图 9 原始及两类扰动输入下的准确率',6.2)
table(['模型','原始','变暗','噪声'],[[CN[n],pct(results[n]['test']['accuracy']),pct(robust[n]['dark']),pct(robust[n]['noise'])] for n in NAMES])
caption('表 8 场景扰动测试结果')

page('十一 分类应用与使用边界')
para('项目提供 predict.py 命令行应用：读取本地图片，将其调整为 32×32 RGB，按训练均值和标准差归一化，加载指定检查点，输出概率最高的三个类别。这样可以把实验模型转化为可运行的分类程序，并在普通 CPU 环境演示。')
figure('predictions.png','图 10 测试集正确识别与错误识别的样例',6.1)
h2('场景选型')
para(f'如果优先追求最高准确率，当前四组中应选择{CN[best["name"]]}；如果更重视计算成本，则应结合表 7 的 MACs 和本机 CPU 延迟选择。所有比较只覆盖同一数据集、同一输入分辨率和同一台测试设备。')
h2('适用范围')
para('CIFAR-10 只有 32×32 像素，来自外部摄像头的照片在清晰度、背景和拍摄条件上可能不同。外部照片可作定性演示，但没有带标注的独立真实场景样本时，不能宣称该模型在真实摄像头环境达到 CIFAR-10 测试准确率。')

page('十二 运行说明与结果核验')
para(f'代码仓库：{repo_url}')
para('项目根目录包含 model.py、train.py、analyze.py、predict.py、requirements.txt、README.md、data、results 和 figures。先安装依赖，再解压官方数据集，依次运行四个训练命令。最后运行 analyze.py 重建图表与扰动测试，运行 make_report.py 从结果 JSON 生成本报告。')
table(['步骤','命令'],[['安装依赖','python -m pip install -r requirements.txt'],['解压数据','tar -xzf data/cifar-10-python.tar.gz -C data'],['训练基线','python train.py --model baseline --epochs 30'],['其他三组','将 --model 改为 depthwise、se、depthwise_se'],['生成图表','python analyze.py'],['单图演示','python predict.py 图片路径 --model baseline']])
caption('表 9 从数据到报告的最小复现步骤')
para('每组训练生成一个 .pt 检查点和一个 .json 指标文件。报告中的所有准确率、参数量、MACs、推理时间与训练曲线均从这些结果文件读取；make_report.py 在缺少文件时会报错，而不会填充示例值。')

page('十三 结论 局限与参考资料')
para(f'本实验完成了 CIFAR-10 上的半宽 ResNet-18 基础实现、深度可分离卷积与 SE 注意力两类结构改动及组合消融。基线测试准确率为 {pct(base["test"]["accuracy"])}；轻量化结构参数量减少 {100*(1-dw["parameters"]/base["parameters"]):.2f}%，MACs 减少 {100*(1-dw["macs"]/base["macs"]):.2f}%。是否值得采用某一改动，应以表 5、表 7 和表 8 的实际精度、延迟与扰动表现共同判断。')
para('局限包括仅使用一个随机种子、只评估 CIFAR-10、CPU 延迟只在一台设备上测量，以及扰动测试不能代替真实部署数据验证。后续可增加多种子重复、真实终端测试与额外标注照片。')
h2('参考资料')
for ref in ['[1] Krizhevsky A. CIFAR-10 and CIFAR-100 datasets. https://www.cs.toronto.edu/~kriz/cifar.html',
            '[2] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR 2016.',
            '[3] Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR 2018.',
            '[4] PyTorch. Conv2d and CUDA documentation. https://docs.pytorch.org/']:
    p=doc.add_paragraph(ref);p.paragraph_format.first_line_indent=Pt(0)

footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('CIFAR-10 CNN 结构实验  ·  ')
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
dest=ROOT/'CNN结构修改与场景适配实验报告.docx'
doc.save(dest)
print(dest)
