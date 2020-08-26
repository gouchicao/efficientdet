# EfficientDet

## Google AutoML EfficientDet

### 拉取容器镜像
```bash
docker pull gouchicao/efficientdet:latest
```

### 运行容器
```bash
sudo docker run --runtime=nvidia -it gouchicao/efficientdet:latest bash
```

### 自定义数据集
1. 选择生成的 class
```bash
nano dataset_custom/create_pascal_tfrecord.py

# 注释或者删除变量 pascal_label_map_dict 中不需要的项。
pascal_label_map_dict = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    # 'bird': 3,
    # 'boat': 4,
    # 'bottle': 5,
    # 'bus': 6,
    # 'car': 7,
    # 'cat': 8,
    # 'chair': 9,
    # 'cow': 10,
    # 'diningtable': 11,
    # 'dog': 12,
    # 'horse': 13,
    # 'motorbike': 14,
    # 'person': 15,
    # 'pottedplant': 16,
    # 'sheep': 17,
    # 'sofa': 18,
    # 'train': 19,
    # 'tvmonitor': 20,
}
```

2. 生成 tfrecord
```bash
mkdir tfrecord
PYTHONPATH=".:$PYTHONPATH" python dataset_custom/create_pascal_tfrecord.py \
    --data_dir=/VOCtrainval_11-May-2012/VOCdevkit --year=VOC2012 --output_path=tfrecord/pascal

I0826 17:26:40.648776 140644300003136 create_pascal_tfrecord.py:239] writing to output path: tfrecord/pascal
I0826 17:26:40.651618 140644300003136 create_pascal_tfrecord.py:265] Reading from PASCAL VOC2012 dataset.
I0826 17:26:40.675092 140644300003136 create_pascal_tfrecord.py:277] On image 0 of 595
I0826 17:26:40.751746 140644300003136 create_pascal_tfrecord.py:277] On image 100 of 595
I0826 17:26:40.825970 140644300003136 create_pascal_tfrecord.py:277] On image 200 of 595
I0826 17:26:40.899290 140644300003136 create_pascal_tfrecord.py:277] On image 300 of 595
I0826 17:26:40.969913 140644300003136 create_pascal_tfrecord.py:277] On image 400 of 595
I0826 17:26:41.041234 140644300003136 create_pascal_tfrecord.py:277] On image 500 of 595
```

### 定义使用的预训练模型
```bash
MODEL='efficientdet-d1'
```

### 设置超参数文件 voc_config.yaml
```yaml
num_classes: 3
var_freeze_expr: '(efficientnet|fpn_cells|resample_p6)'
label_map: {1: aeroplane, 2: bicycle}
learning_rate: 0.002
lr_warmup_init: 0.0002
clip_gradients_norm: 5.0
```

### 训练模型
```bash
python main.py --mode=train_and_eval \
    --training_file_pattern=tfrecord/pascal*.tfrecord \
    --validation_file_pattern=tfrecord/pascal*.tfrecord \
    --model_name=$MODEL \
    --model_dir=/tmp/$MODEL-finetune  \
    --ckpt=$MODEL  \
    --train_batch_size=12 \
    --eval_batch_size=12 --eval_samples=36 \
    --num_examples_per_epoch=595 --num_epochs=100  \
    --hparams=voc_config.yaml
```

### 预测
```bash
python model_inspect.py --runmode=infer \
    --model_name=$MODEL　--ckpt_path=/tmp/$MODEL-finetune \
    --hparams=voc_config.yaml  \
    --input_image=test.jpg --output_image_dir=/tmp/
```
