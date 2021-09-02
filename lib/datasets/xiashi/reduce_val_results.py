# System libs
import os
import time
import argparse
import pickle
# Our libs
import pandas as pd
import numpy as np


from lib.datasets.xiashi.default import _C as cfg
from collections import defaultdict

import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(cur_path, ".."))

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/id-hrnet.yaml",  # "config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--gpus",
        default="0,1",  # "0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )

    parser.add_argument(
        "--export",
        action='store_true',
        help="store true to export valid info"
    )

    parser.add_argument(
        "--type",
        default="no-split",  # "0-3",
        help="the testing type"
    )

    return parser.parse_args()


def parse_csv_merged(cfg):
    colors_csv = pd.read_csv(cfg.DATASET.csv_file)
    _, ind = np.unique(colors_csv.iloc[:, -1].to_numpy(), return_index=True)
    colors = colors_csv.iloc[ind, 7: 10].values.astype(dtype=np.uint8)
    class_names = colors_csv.iloc[ind, 6].values
    return colors, class_names


def setup_logger(name, distributed_rank, save_dir=None, filename="log.txt", mode='w'):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



def parseIntSet(nputstr=""):
    """
    https://stackoverflow.com/questions/712460/interpreting-number-ranges-in-python
    :param nputstr:
    :return:
    """
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        if len(i) > 0:
            if i[:1] == "<":
                i = "1-%s"%(i[1:])
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                # token = [int(k.strip()) for k in i.split('-')]
                token = [int(k.strip()) for k in i.split('~')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token)-1]
                    for x in range(first, last+1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        print("Invalid set: " + str(invalid))
    return list(selection)
# end parseIntSet


def load_results(epoch):
    colors, class_names = parse_csv_merged(cfg)

    if cfg.VAL.use_split:
        path_name = os.path.join(cfg.DIR, 'dist_split_result_size{}'.format(cfg.DATASET.imgValMaxSize), str(epoch))
    else:
        path_name = os.path.join(cfg.DIR, 'dist_result_size{}'.format(cfg.DATASET.imgValMaxSize), str(epoch))

    logger.info("open {}".format(path_name))
    assert os.path.exists(path_name)
    # reduce_results = defaultdict(int)

    intersection = np.zeros(cfg.DATASET.num_class)
    union = np.zeros(cfg.DATASET.num_class)
    acc = 0
    time = 0
    num = 0
    pixel_count = 0
    for name in os.listdir(path_name):
        logger.info("reduce {}/{}".format(path_name,name))
        with open(os.path.join(path_name, name), 'rb') as f:
            save_dict = pickle.load(f)
        intersection += save_dict['inter']
        union += save_dict['uni']
        acc += save_dict['pixel_acc']
        time += save_dict["average_time"]
        num += save_dict['num']
        pixel_count += save_dict["pixel_count"]

    iou = intersection / (union + 1e-10)
    acc = acc/pixel_count
    time = time/num
    logger.info("reduced results for epoch {}".format(epoch))
    # for i,k in enumerate(reduce_results.keys()):
    #     reduce_results[k] = reduce_results[k]/2.0
    #     logger.info('[{}] {}: {:.4f}'.format(i, k, reduce_results[k]))

    for i, _iou in enumerate(iou):
        logger.info('class [{}] {}, IoU: {:.4f}'.format(i, class_names[i], _iou))

    logger.info('Reduced Result : Mean IOU (except_ignore): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou[:-1].mean(), iou.mean(), acc, time))


def load_results_per_height(epoch, if_export, test_type):
    colors, class_names = parse_csv_merged(cfg)

    if if_export:
        fp = open("./reduced_valid_info.log", "a+")

    if cfg.VAL.use_split:
        path_name = os.path.join(cfg.DIR, 'dist_split_result_size{}'.format(cfg.DATASET.imgValMaxSize), str(epoch))
    else:
        path_name = os.path.join(cfg.DIR, 'dist_result_size{}'.format(cfg.DATASET.imgValMaxSize), str(epoch))

    logger.info("reduce results for epoch {}".format(epoch))
    logger.info("open {}".format(path_name))
    assert os.path.exists(path_name)
    # reduce_results = defaultdict(int)

    height_keys = ['all', 30, 60, 90, 120]
    intersection_dict = {}
    union_dict = {}
    for k in height_keys:
        intersection_dict[k] = np.zeros(cfg.DATASET.num_class)
        union_dict[k] = np.zeros(cfg.DATASET.num_class)

    acc_dict = defaultdict(float)
    # time = defaultdict(float)
    num_dict = defaultdict(int)
    pixel_count_dict = defaultdict(int)

    for name in os.listdir(path_name):
        logger.info("reduce {}/{}".format(path_name,name))
        with open(os.path.join(path_name, name), 'rb') as f:
            save_dict = pickle.load(f)
        for k in save_dict.keys():
            # print(k, save_dict[k])
            intersection_dict[k] += save_dict[k]['inter']
            union_dict[k] += save_dict[k]['uni']

            acc_dict[k] += save_dict[k]['pixel_acc']
            # time += save_dict["average_time"]
            num_dict[k] += save_dict[k]['num']
            pixel_count_dict[k] += save_dict[k]["pixel_count"]

    if if_export:
        # import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')

        print("------ The testing type is: {} ------".format(test_type), file=fp, flush=True)
        print("------ Current time is: {} ------".format(time_str), file=fp, flush=True)

    for k in save_dict.keys():
        logger.info(
            "##################################################################################")
        logger.info("reduced results for dataset at {} height for epoch {}".format(k, epoch))
        intersection = intersection_dict[k]
        union = union_dict[k]
        acc = acc_dict[k]
        pixel_count = pixel_count_dict[k]
        num = num_dict[k]

        iou = intersection / (union + 1e-10)
        acc = acc/(pixel_count+1e-10)
        # time = time/num

        # for i,k in enumerate(reduce_results.keys()):
        #     reduce_results[k] = reduce_results[k]/2.0
        #     logger.info('[{}] {}: {:.4f}'.format(i, k, reduce_results[k]))

        for i, _iou in enumerate(iou):
            logger.info('class [{}] {}, IoU: {:.4f}'.format(i, class_names[i], _iou))
        logger.info(
            'Reduced Result : Mean IOU (except_ignore & people): {:.4f}, Mean IOU (stuff): {:.4f}, Mean IOU (object): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%'
            .format(iou[:-2].mean(), iou[:10].mean(), iou[10:-2].mean(), iou.mean(), acc))

        if if_export:
            print("reduced results for dataset at {} height for epoch {}".format(k, epoch), file=fp, flush=True)
            print(
                'Reduced Result : Mean IOU (except_ignore & people): {:.4f}, Mean IOU (stuff): {:.4f}, Mean IOU (object): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%'
                .format(iou[:-2].mean(), iou[:10].mean(), iou[10:-2].mean(), iou.mean(), acc), file=fp, flush=True)

        # logger.info('Reduced Result : Mean IOU (except_ignore): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
        #       .format(iou[:-1].mean(), iou.mean(), acc, time))


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg)
    if len(cfg.VAL.checkpoint) == 0:
        if cfg.TRAIN.data_crop:
            cfg.DIR = './'
            # cfg.DIR = '{}/{}_{}_{}/{}_{}_class{}_size{}'.format(args.ckpt_path,cfg.DATASET.train_data_name,cfg.DATASET.crop_type, cfg.MODEL.arch,
            #                                                            cfg.MODEL.arch_encoder,
            #                                                            cfg.MODEL.arch_decoder, cfg.DATASET.num_class,
            #                                                            cfg.DATASET.imgMaxSize)
        else:
            cfg.DIR = './'
            # cfg.DIR = '{}/{}_{}/{}_{}_class{}_size{}'.format(args.ckpt_path, cfg.DATASET.train_data_name,
            #                                                  cfg.MODEL.arch,
            #                                                  cfg.MODEL.arch_encoder,
            #                                                  cfg.MODEL.arch_decoder, cfg.DATASET.num_class,
            #                                                  cfg.DATASET.imgMaxSize)
    else:
        checkpoint_name = os.path.basename(cfg.VAL.checkpoint).replace('.pth', '')
        cfg.DIR = 'val_results/{}'.format(checkpoint_name)


    if "ocr" in cfg.MODEL.arch and cfg.OCR.if_object_mask:
        cfg.DIR = cfg.DIR + '_objectmask'
    if cfg.TRAIN.use_ohem:
        cfg.DIR = cfg.DIR + '_ohem'
    if cfg.TRAIN.se_loss:
        cfg.DIR = cfg.DIR + '_se'
    if cfg.TRAIN.metric_loss:
        cfg.DIR = cfg.DIR + '_metric'

    epochs = parseIntSet(cfg.VAL.multi_checkpoints)
    epochs = [sorted(epochs)[-1]]
    import datetime
    ISOTIMEFORMAT = '%Y-%m-%d-%H:%M:%S'
    if cfg.VAL.use_split:
        if len(cfg.VAL.checkpoint) == 0:
            log_filename = '{}'.format(cfg.DATASET.val_data_name) + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) \
                           + '_reduced_val_split_' + 'epoch{}_{}'.format(epochs[0], epochs[-1]) + '.txt'
        else:
            log_filename = '{}'.format(cfg.DATASET.val_data_name) + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) \
                           + '_reduced_val_split' + '.txt'

    else:
        if len(cfg.VAL.checkpoint) == 0:
            log_filename = '{}'.format("xiashi") + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) \
                           + '_reduced_val_' + 'epoch{}_{}'.format(epochs[0], epochs[-1]) + '.txt'
        else:
            log_filename = '{}'.format("xiashi")  + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) \
                           + '_reduced_val' + '.txt'


    logger = setup_logger(name='Logger', distributed_rank=0, save_dir=cfg.DIR,
                          filename=log_filename,
                          mode='w')
    logger.info(
        "######################################### validation summary #########################################")
    logger.info("saving reduced results to {}".format(log_filename))
    print("Whether to export valid info to file: {}".format(args.export))
    if len(cfg.VAL.checkpoint) == 0:
        for epoch in epochs:
            # load_results(epoch)
            print("--- valid for epoch {} ---".format(epoch))
            load_results_per_height(epoch, args.export, args.type)
    else:
        load_results_per_height(os.path.basename(cfg.VAL.checkpoint).replace('.pth',''))


