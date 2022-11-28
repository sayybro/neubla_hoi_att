python -u vis_demo.py \
    --pretrained checkpoints/hoi/checkpoint0059.pth \
    --mtl \
    --show_vid \
    --update_obj_att \
    --num_obj_att_classes 80 \
    --num_obj_classes 81 \
    --num_verb_classes 117 \
    --mtl_data [\'vcoco\',\'hico\'] \
    --backbone resnet50 \
    --eval \
    ${PY_ARGS}