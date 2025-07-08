CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py --batch-size 64 -a resnet50 -d market1501 --epochs 1 --iters 10 --momentum 0.1 --eps 0.6 --num-instances 16 --eps-gap 0.0 --k1 15 --k2 4



#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16


# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224
