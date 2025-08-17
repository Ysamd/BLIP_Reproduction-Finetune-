# BLIP的复现微调指南:smile:
分为以下几个步骤:
- 下载源代码和预训练权重
- 下载数据集
- 配置环境依赖
- 利用官方预训练权重进行微调
- 微调结果与论文进行对比
- 其他问题
下面详细说明:

## 下载源代码和预训练权重

进入BLIP的[官方repo](https://github.com/salesforce/BLIP)获取源代码和预训练权重

## 下载数据集
[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html):打开后下载前两个即可，即图像和标注文件

[COCO](https://cocodataset.org/#download):在Retrieval以及Caption用的都是COCO2014的图像，即只要2014的Train和Val的图像和annotations

[VQAv2](https://visualqa.org/download.html):这里用的image和上边一样都是COCO，但是要多一个test2015，需要在VQA官网上下载的有[Training annotations 2017 v2.0*](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip)，[Validation annotations 2017 v2.0*](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)，  
[Training questions 2017 v2.0*](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)，[Validation questions 2017 v2.0*](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)，[Testing questions 2017 v2.0](https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)，注意这里test是没有答案的

[NLVR2](https://lil.nlp.cornell.edu/resources/NLVR2/):这里我只下载了前三个，如果你想提交结果，可以下载第四个，至于标注文件，在NLVR的[官方repo](https://github.com/lil-lab/nlvr/tree/master/nlvr2)里可以找到，但要自行做一些处理，不然跑不动

## 配置环境依赖
创建新环境:
```python
conda create -n BLIP python==3.9
```

下载要用的包:
```python
pip install timm==0.4.12
pip install transformers==4.15.0
pip install fairscale==0.4.4
pip install pycocoevalcap
```
## 利用官方预训练权重进行微调
这里以VQA为例(运行命令都大同小异)：
```python
python VQA.py \
  --config ./configs/VQA.yaml \
  --output_dir ./output/vqa \
  --device cuda \
  --evaluate
```

## 与官方论文进行对比
BLIP的官方论文[地址](https://arxiv.org/pdf/2201.12086)
利用得到的日志文件进行对比

## 其他问题
- 如果遇到OOM的问题，请在配置文件(configs文件夹中)找到对应的下游任务，修改batch size和图片分辨率
- 配置文件中，图像的路径需要回退一级，最后一级是作者已经写死的，不需要加在配置文件中，不然会检索失败
- 如果你用的是单卡训练，需要在训练的py文件中这么做：
```python
if args.distributed:
	dist.barrier() ##这一行是固有的，需要做一个判断条件
```
- 如果你发现全量微调Loss不下降，可以先冻结backbone，先训练分类头，等收敛后再逐步解冻

