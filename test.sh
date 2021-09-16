DATA_ROOT="/home/hadoop-automl/cephfs/data/zhaowangbo/dataset/cityscapes2"
SCRATCH_ROOT="output"
ASSET_ROOT=${DATA_ROOT}
MODEL_ROOT="/home/hadoop-automl/cephfs/data/linxinyang/models"

DATA_DIR="${DATA_ROOT}"
SAVE_DIR="${SCRATCH_ROOT}/xiashi_seg_results/"


BACKBONE="hrnet48"

CONFIGS="configs/meituan_xiashi/H_48_D_4.json"
MODEL_NAME="hrnet_w48_ocr_contrast"
LOSS_TYPE="contrast_auxce_loss"

CHECKPOINTS_ROOT="${SCRATCH_ROOT}/Cityscapes/"
CHECKPOINTS_NAME="${MODEL_NAME}_lr1x_0825"
PRETRAINED_MODEL="${MODEL_ROOT}/hrnetv2_w48_imagenet_pretrained.pth"
LOG_FILE="${SCRATCH_ROOT}/logs/Cityscapes/${CHECKPOINTS_NAME}.log"
MAX_ITERS=20000
#MAX_ITERS=200
BATCH_SIZE=16
BASE_LR=0.01

# --gpu 0 1 2 3 4 5 6 7 \

python -u main_contrastive.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase test \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 5 \
                       --use_xiashi_dataset \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --train_batch_size ${BATCH_SIZE} \
                       --distributed \
                       --base_lr ${BASE_LR} \
                       2>&1 | tee ${LOG_FILE}


export CFG_1="configs/meituan_xiashi/mergev1.4_val_nosplit.yaml"
export CFG_2="configs/meituan_xiashi/mergev1.4_val_splitA.yaml"
export GPUS=0,1,2,3,4,5,6,7
echo "-----------------------------------------------"
echo "Start to valid no-split experiment"
echo "-----------------------------------------------"
python lib/datasets/xiashi/reduce_val_results.py --cfg=$CFG_1 --gpus=$GPUS --export --type "no_split"

echo "-----------------------------------------------"
echo "Start to valid split-A experiment"
echo "-----------------------------------------------"
python lib/datasets/xiashi/reduce_val_results.py --cfg=$CFG_2 --gpus=$GPUS --export --type "split-A"
