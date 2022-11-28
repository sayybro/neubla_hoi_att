# Training

## For single task training
```
CUDA_VISIBLE_DEVICES=1,2 GPUS_PER_NODE=2 ./tool/run_dist_launch.sh 2 configs/mtl_train.sh \
        --mtl_data [\'vaw\'] \
        --output_dir checkpoints/vaw \
        --pretrained params/detr-r50-pre-vaw.pth
```  

## For multi task training
```
CUDA_VISIBLE_DEVICES=1,2 GPUS_PER_NODE=2 ./tool/run_dist_launch.sh 2 configs/mtl_train.sh \
        --mtl_data [\'vcoco\',\'hico\',\'vaw\'] \
        --output_dir checkpoints/mtl_all.pth \
        --pretrained params/detr-r50-pre-mtl.pth
``` 

# Multi task learning evaluation
## vcoco evaluation
```
"test_mAP_all": 0.5459505229340162, "test_mAP_thesis": 0.5670662778460144
```
## hico evaluation
```
"test_mAP": 0.2789136106056413, "test_mAP rare": 0.20482938557040611, "test_mAP non-rare": 0.3010426648369453
```
## vaw evaluation
```
"test_mAP": 0.05324181350284175, "test_mAP rare": 0.03170080627445978, "test_mAP non-rare": 0.07084839018722701
```

# Video demo inference
## For vcoco verb inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type vcoco \
        --mtl_data [\'vcoco\'] \
        --mtl \
        --video_file video/cycle.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4
```  
## For hico verb inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type hico \
        --mtl_data [\'hico\'] \
        --mtl \
        --video_file video/cycle.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4
```  
## For vaw attribute inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type vaw \
        --mtl_data [\'vaw\'] \
        --mtl \
        --video_file video/animal.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4
```  





## Citation
Please consider citing our paper if it helps your research.
```
@inproceedings{tamura_cvpr2021,
author = {Tamura, Masato and Ohashi, Hiroki and Yoshinaga, Tomoaki},
title = {{QPIC}: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information},
booktitle={CVPR},
year = {2021},
}
```
