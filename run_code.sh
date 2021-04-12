CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.4 --num-instances 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d market1501 --iters 200 --momentum 0.1 --eps 0.4 --num-instances 16

#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --height 224 --width 224
#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d veri --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --height 224 --width 224

#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d personx --iters 200 --momentum 0.1 --eps 0.7 --num-instances 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d personx --iters 200 --momentum 0.1 --eps 0.7 --num-instances 16

#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.7 --num-instances 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16

#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d msmt17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16


#sudo nvidia-docker run --shm-size 20G -it -v /data1/yixuan/data/person_reid:/workspace/SpCL/examples/data reg.docker.alibaba-inc.com/virtualbuy/spcl:v1
