IMAGENET_FOLD=C:/Users/furka/Desktop/ki/ILSVRC2012_img_train_t3

GPU=${1} # gpu used
CHECKPOINTS_FOLD=${2} #path_to_output_folder

#python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                      --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10
python JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --batch 128 --lr=0.001 --cores=10
