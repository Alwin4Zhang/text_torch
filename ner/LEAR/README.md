# LEAR

The implementation our EMNLP 2021 paper ["Enhanced Language Representation with Label Knowledge for Span Extraction"](https://arxiv.org/pdf/2111.00884.pdf).

See below for an overview of the model architecture:

![Untitled](aaa%20becd43cc06bf4820ba2e6bfe3a780420/Untitled.png)

## Requirements

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Data

### pretrained model

[https://huggingface.co/hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)

```bash
git lfs install
git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

### dataset

下载对应文件并解压到根目录的data目录下

[https://cloud.189.cn/t/NjUrauVfum2m（访问码：4xd0）](https://cloud.189.cn/t/NjUrauVfum2m%EF%BC%88%E8%AE%BF%E9%97%AE%E7%A0%81%EF%BC%9A4xd0%EF%BC%89)

## Train

```bash
# MSRA 数据集
CUDA_VISIBLE_DEVICES=0 python run_ner.py --task_type sequence_classification --task_save_name SERS --data_dir ./data/ner --data_name zh_msra --model_name SERS --model_name_or_path ./pretrained/chinese-roberta-wwm-ext-large --output_dir ./zh_msra_models/bert_large --do_lower_case False --result_dir ./zh_msra_models/results --first_label_file ./data/ner/zh_msra/processed/label_map.json --overwrite_output_dir True --train_set ./data/ner/zh_msra/processed/train.json --dev_set ./data/ner/zh_msra/processed/dev.json --test_set ./data/ner/zh_msra/processed/test.json --is_chinese True --max_seq_length 128 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 5 --learning_rate 8e-6 --task_layer_lr 10 --label_str_file ./data/ner/zh_msra/processed/label_annotation.txt --span_decode_strategy v5

# onto4 数据集
CUDA_VISIBLE_DEVICES=0 python run_ner.py --task_type sequence_classification --task_save_name SERS --data_dir ./data/ner --data_name zh_onto4 --model_name SERS --model_name_or_path ./pretrained/chinese-roberta-wwm-ext-large --output_dir ./zh_onto4_models/bert_large --do_lower_case False --result_dir ./zh_onto4_models/results --first_label_file ./data/ner/zh_onto4/processed/label_map.json --overwrite_output_dir True --train_set ./data/ner/zh_onto4/processed/train.json --dev_set ./data/ner/zh_onto4/processed/dev.json --test_set ./data/ner/zh_onto4/processed/test.json --is_chinese True --max_seq_length 128 --per_gpu_train_batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 5 --learning_rate 8e-6 --task_layer_lr 10 --label_str_file ./data/ner/zh_onto4/processed/label_annotation.txt --span_decode_strategy v5
```

### 环境:

```bash
Driver Version:470.129.06
CUDA Version: 10.2
GPU: rtx3080 10G
CPU: 8c16t Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
RAM: 32G
```

| 数据集 | best f1/p/r |
| --- | --- |
| onto4 | f1: 0.8245, p: 0.8345(6186/7413), r: 0.8148(6186/7592) |
| msra | f1: 0.9558, p: 0.9620(5271/5479), r: 0.9496(5271/5551) |

## Evaluation

- `Nested: set `exist_nested`=True.`
- `Flat: set `span_decode_strategy`=v5.`


Last edited time: October 18, 2022 3:23 PM  
Property: October 18, 2022 3:23 PM  
Status: Done  