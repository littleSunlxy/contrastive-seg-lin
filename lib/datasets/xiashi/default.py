from yacs.config import CfgNode as CN
import numpy as np
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"
_C.DIR_TEST = "ckpt/ade20k-resnet50dilated-ppm_deepsup-test"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.datatype = "xiashidata"
_C.DATASET.root_dataset = "/workdir/hrr/data"
_C.DATASET.list_train = "/workdir/hrr/data/training.odgt"
_C.DATASET.list_val = "/workdir/hrr/data/validation.odgt"
_C.DATASET.list_test = "/workdir/hrr/data/test.odgt"
_C.DATASET.csv_file = '/workdir/hrr/data/class.csv'
_C.DATASET.num_class = 150
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgValSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgValMaxSize = 1000
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgTestSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgTestMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
_C.DATASET.train_data_name = ''
_C.DATASET.val_data_name = ''
_C.DATASET.crop_type = 'input_normal'
_C.DATASET.spilt_mode = "all"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048

_C.MODEL.arch = "hardnet"
_C.MODEL.use_teacher = False
#deeplabv3plus
_C.MODEL.aspp_global_feature = False

# cityscapes from HRNet codebase
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 2



_C.OCR = CN()
_C.OCR.if_object_mask = False

_C.MSCALE = CN()
_C.MSCALE.scale_list = (0.5, 1, 1.5)
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.use_fp16 = False
_C.TRAIN.use_fp16_O2 = False
_C.TRAIN.use_ohem = False
_C.TRAIN.use_seesaw = False
_C.TRAIN.use_Weightedlovasz = False
_C.TRAIN.aux = False
_C.TRAIN.aux_weight = 0.4
_C.TRAIN.se_loss = False
_C.TRAIN.se_weight = 0.2
_C.TRAIN.metric_loss = False
_C.TRAIN.metric_weight = 0.001
_C.TRAIN.use_reweight = 0
_C.TRAIN.data_crop = False
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# load pretrained path, default is the same to model save path
_C.TRAIN.pretrain_is_imagenet = True
_C.TRAIN.pretrained_model_path = 'pretrained/resnet18-imagenet.pth'
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304

_C.TRAIN.warmup_steps = 1000
_C.TRAIN.warmup_start_lr = 1e-5

_C.TRAIN.load_checkpoint = ''

_C.TRAIN.monitor = True
_C.TRAIN.ckpt_dir = ''

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.LOSS = CN()
_C.LOSS.OCR_AUX_RMI = False
#_C.LOSS.OCR_AUX_RMI = True


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
_C.VAL.val_height = -1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = ""
_C.VAL.multi_checkpoints = "1~10"
# number of data loading workers
_C.VAL.workers = 5
# _C.VAL.img_output_scale = 1
_C.VAL.save_labelfile = False
_C.VAL.use_split = False

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
_C.TEST.workers = 0
# the checkpoint to test on
# _C.TEST.checkpoint = "epoch_20.pth"
_C.TEST.checkpoint = "epoch_20.pth"
_C.TEST.multi_checkpoints = "1~10"
# folder to output visualization results
_C.TEST.vis_result = "./"
_C.TEST.save_labelfile = False
_C.TEST.visualize = False
_C.TEST.img_split = 1
_C.TEST.imgSizes = (300, 375, 450, 525, 600)
_C.TEST.imgMaxSize = 1000
_C.TEST.use_split = False
_C.TEST.visualize_onlypredict = False


# cityscapes from HRNet codebase
_C.TEST.OUTPUT_INDEX = -1


def cfg_from_file(filename, args):
    """Load a config file and merge it into the default options."""
    _C.merge_from_file(filename)
    _C.merge_from_list(args.opts)

