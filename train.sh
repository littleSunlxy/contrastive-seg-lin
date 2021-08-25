DATA_ROOT="/home/hadoop-automl/cephfs/data/linxinyang/dataset/cityscapes"
SCRATCH_ROOT="output"
ASSET_ROOT=${DATA_ROOT}
MODEL_ROOT="/home/hadoop-automl/cephfs/data/linxinyang/models"

DATA_DIR="${DATA_ROOT}/Cityscapes"
SAVE_DIR="${SCRATCH_ROOT}/cityscapes_seg_results/"

BACKBONE="hrnet48"

CONFIGS="configs/cityscapes/H_48_D_4.json"
MODEL_NAME="hrnet_w48_ocr_contrast"
LOSS_TYPE="contrast_auxce_loss"

CHECKPOINTS_ROOT="${SCRATCH_ROOT}/Cityscapes/"
CHECKPOINTS_NAME="${MODEL_NAME}_lr1x_0825"
PRETRAINED_MODEL="${MODEL_ROOT}/hrnetv2_w48_imagenet_pretrained.pth"
LOG_FILE="${SCRATCH_ROOT}/logs/Cityscapes/${CHECKPOINTS_NAME}.log"
MAX_ITERS=40000
BATCH_SIZE=2
BASE_LR=0.01


python -u main_contrastive.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 1 \
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