python setup.py develop
#CUDA_VISIBLE_DEVICES=0,1,2,3 python basicsr/train.py -opt options/train_FeMaSR_LQ_stage.yml
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 basicsr/train.py -opt options/train_FeMaSR_LQ_stage.yml --launcher pytorch