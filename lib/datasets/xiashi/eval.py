from lib.utils import as_numpy
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import torch.distributed as dist
import pandas as pd
import pickle
import cv2
from PIL import Image

from lib.datasets.xiashi.dataset_cv import  user_scattered_collate,ValDataset_cv, ValDataset_Split_cv
from lib.datasets.xiashi.utils import extract_height, AverageMeter, colorEncode, parse_devices, accuracy, intersectionAndUnion, \
    setup_logger, parseIntSet


iou_mean = []
iou_mean_except_ignore = []
acc_mean = []
inference_mean = []


def parse_csv_merged(cfg):
    colors_csv = pd.read_csv(cfg.DATASET.csv_file)
    _, ind = np.unique(colors_csv.iloc[:, -1].to_numpy(), return_index=True)
    colors = colors_csv.iloc[ind, 7: 10].values.astype(dtype=np.uint8)
    class_names = colors_csv.iloc[ind, 6].values
    # ['lawn' 'Flat_ground' 'Ground' 'Roof' 'Plant' 'Road' 'Sports_ground'
    # 'Water' 'Building ' 'Vehicle ' 'Ship ' 'people' 'Barrier '
    # 'High_barrier ' 'High_tension ' 'Sky' 'Ignore ']
    #     print(colors, class_names)
    return colors, class_names


def save_labelfile(info, pred, dir_result):
    img_name = os.path.basename(info).rpartition('.')[0]
    Image.fromarray(pred.astype(np.int32)).convert('L').save(os.path.join(dir_result, img_name + '.png'))



def visualize_result(data, pred, dir_result, colors):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    img = Image.fromarray(img)
    img = img.resize((seg_color.shape[1], seg_color.shape[0]), Image.BILINEAR)
    img = np.array(img)
    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))
    ##### save blended images
    img_blended = cv2.addWeighted(img, 0.7, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR), 1 - 0.7, 0)
    if not os.path.exists(os.path.join(dir_result, 'blended')):
        os.makedirs(os.path.join(dir_result, 'blended'))
    cv2.imwrite(
        os.path.join(dir_result, 'blended', img_name.replace('.jpg', '-blended.png').replace('.JPG', '-blended.png')),
        img_blended)



def evaluate(model, loader, epoch):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()
    all_time_meter = AverageMeter()
    data_time = AverageMeter()

    acc_meter_30m = AverageMeter()
    intersection_meter_30m = AverageMeter()
    union_meter_30m = AverageMeter()

    acc_meter_60m = AverageMeter()
    intersection_meter_60m = AverageMeter()
    union_meter_60m = AverageMeter()

    acc_meter_90m = AverageMeter()
    intersection_meter_90m = AverageMeter()
    union_meter_90m = AverageMeter()

    acc_meter_120m = AverageMeter()
    intersection_meter_120m = AverageMeter()
    union_meter_120m = AverageMeter()

    acc_meter_per_height = {30: acc_meter_30m, 60: acc_meter_60m, 90: acc_meter_90m, 120: acc_meter_120m}
    intersection_meter_per_height = {30: intersection_meter_30m, 60: intersection_meter_60m, 90: intersection_meter_90m,
                                     120: intersection_meter_120m}
    union_meter_per_height = {30: union_meter_30m, 60: union_meter_60m, 90: union_meter_90m, 120: union_meter_120m}

    from _collections import defaultdict
    num_per_height = defaultdict(int)

    # colors, class_names = parse_csv(cfg)
    colors, class_names = parse_csv_merged(cfg)

    model.eval()
    save_path = epoch
    logger.info('[Epoch {} Eval Summary]:'.format(epoch))
    # pbar = tqdm(total=len(loader))
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # process data
        # if i>3:
        #     break
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        if cfg.TRAIN.use_reweight == 3:
            mask = (seg_label != 11) * (seg_label != 12) * (seg_label != 13) * (seg_label != -1)
            seg_label[mask] = 0
        #             print(seg_label)
        img_resized_list = batch_data['img_data']

        ### extract height
        # import pdb; pdb.set_trace()
        height = extract_height(os.path.dirname(batch_data['info']))
        # image_name = batch_data['info'].split('/')[-2]
        # index_1 = image_name.find(')')
        # if index_1 != -1:
        #     height = int(image_name[index_1 + 1:])
        # else:
        #     height = -1

        num_per_height[height] += 1

        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)


        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]).cuda()
            # scores = async_copy_to(scores, gpu)

            for index, img in enumerate(img_resized_list):
                img = img.cuda()
                # print(img.size())

                if cfg.MODEL.arch == "hrnet_ocr_v1":
                    # import pdb
                    # pdb.set_trace()
                    torch.cuda.synchronize()
                    start = time.time()
                    scores_tmp = model(img, batch_data['seg_label'])[0]  # <class 'tuple'> 2
                    torch.cuda.synchronize()
                    end = time.time()
                    print("nospilt inference time:", end - start)
                    exit()
                    # import pdb
                    # pdb.set_trace()

                elif cfg.MODEL.arch == "hrnet_ocr.HRNet_Mscale":
                    from config import cfg_nv as cfg_local
                    if not cfg_local.MODEL.N_SCALES:
                        inputs = {'images': img, 'gts': seg_label[index]}
                        # output_dict = model(inputs)
                        #
                        # _pred = output_dict['pred']
                        scores_tmp = model(inputs)
                    else:
                        inputs = {'images': img, 'gts': seg_label[index]}
                        # output_dict = model(inputs)
                        #
                        # _pred = output_dict['pred']
                        scores_tmp = model(inputs)
                        print(type(scores_tmp), scores_tmp)

                else:
                    scores_tmp = model(img)



                scores_tmp = nn.functional.interpolate(
                    scores_tmp, size=segSize, mode='bilinear', align_corners=False)

                scores_tmp = nn.functional.softmax(scores_tmp, dim=1)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
            #             scores[:, -1, :, :] = 0  # ignore the last category

            # name = batch_data['info'].replace('/', '_').split('.')[0]
            # # add plot code
            # from lib.traintools.visualize import save_pred
            # save_pred(scores, "img/bestmodelpred/", name)
            # exit()

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        class_list = range(cfg.DATASET.num_class)
        #         print(class_list[-2:])
        acc, pix = accuracy(pred, seg_label, ignored_label=class_list[-2:])  # omit ignored label, people
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        # print(intersection, union)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        if height != -1:
            acc_meter_per_height[height].update(acc, pix)
            intersection_meter_per_height[height].update(intersection)
            union_meter_per_height[height].update(union)

        # logger.info()
        # visualization
        if cfg.VAL.visualize:
            # print(os.path.join(cfg.DIR,
            #                    '{}_val_result_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
            #                                                            cfg.DATASET.imgValMaxSize, str(save_path),
            #                                                            height)))
            if not os.path.isdir(os.path.join(cfg.DIR,
                                              '{}_val_result_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
                                                                                      cfg.DATASET.imgValMaxSize,
                                                                                      str(save_path), height))):
                os.makedirs(os.path.join(cfg.DIR,
                                         '{}_val_result_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
                                                                                 cfg.DATASET.imgValMaxSize,
                                                                                 str(save_path),
                                                                                 height)))

            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR,
                             '{}_val_result_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
                                                                     cfg.DATASET.imgValMaxSize, str(save_path),
                                                                     height)),
                colors
            )

        if cfg.VAL.save_labelfile:
            if not os.path.isdir(
                    os.path.join(cfg.DIR, '{}_pred_labelfile_valsize{}'.format(cfg.DATASET.val_data_name,
                                                                               cfg.DATASET.imgValMaxSize),
                                 str(save_path))):
                os.makedirs(
                    os.path.join(cfg.DIR, '{}_pred_labelfile_valsize{}'.format(cfg.DATASET.val_data_name,
                                                                               cfg.DATASET.imgValMaxSize),
                                 str(save_path)))

            save_labelfile(
                batch_data['info'],
                pred,
                os.path.join(cfg.DIR,
                             '{}_pred_labelfile_valsize{}'.format(cfg.DATASET.val_data_name, cfg.DATASET.imgValMaxSize),
                             str(save_path))
            )

        # pbar.update(1)
        # pbar.close()
        # summary

        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        all_time_meter.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        logger.info(
            'Rank[{}] Epoch: [{}][{}/{}] Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Data Time: {:.4f}s, Inference Time: {:.4f}s, Total Time: {:.4f}s'
                .format(dist.get_rank(), epoch, i, len(loader), iou[:-2].mean(), iou.mean(), acc_meter.average() * 100,
                        data_time.average(), time_meter.average(), all_time_meter.average()))
    # torch.cuda.synchronize()

    for i, _iou in enumerate(iou):
        logger.info('Rank[{}] class [{}] {}, IoU: {:.4f}'.format(dist.get_rank(), i, class_names[i], _iou))

        # save_dict[class_names[i]] = _iou

    logger.info(
        'Final Result : Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Data Time: {:.4f}s, Inference Time: {:.4f}s, Total Time: {:.4f}s'
            .format(iou[:-2].mean(), iou.mean(), acc_meter.average() * 100, data_time.average(), time_meter.average(),
                    all_time_meter.average()))

    import pickle
    # save_dict = {}
    save_dict = defaultdict(dict)
    save_dict['all']['inter'] = intersection_meter.sum
    save_dict['all']['uni'] = union_meter.sum
    # print(acc_meter.average(), acc_meter.sum, len(loader), acc_meter.count)
    # print(time_meter.average(), time_meter.sum, time_meter.count)
    save_dict['all']["pixel_acc"] = acc_meter.sum * 100
    # save_dict['all']["average_time"] = time_meter.sum
    save_dict['all']["num"] = len(loader)
    save_dict['all']["pixel_count"] = acc_meter.count

    for h in [30, 60, 90, 120]:
        save_dict[h]['inter'] = intersection_meter_per_height[h].sum
        save_dict[h]['uni'] = union_meter_per_height[h].sum
        # print(acc_meter.average(), acc_meter.sum, len(loader), acc_meter.count)
        # print(time_meter.average(), time_meter.sum, time_meter.count)
        save_dict[h]["pixel_acc"] = acc_meter_per_height[h].sum * 100
        # save_dict[h]["average_time"] = time_meter.sum
        save_dict[h]["num"] = num_per_height[h]
        save_dict[h]["pixel_count"] = acc_meter_per_height[h].count

    path_name = os.path.join(cfg.DIR, 'dist_result_size{}'.format(cfg.DATASET.imgValMaxSize), str(save_path))
    if not os.path.isdir(path_name):
        os.makedirs(path_name)
    with open(os.path.join(path_name, 'dist_rank{}.pkl'.format(dist.get_rank())), 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    torch.cuda.synchronize()
    iou_mean.append(iou.mean())
    iou_mean_except_ignore.append(iou[:-2].mean())
    acc_mean.append(acc_meter.average() * 100)
    inference_mean.append(time_meter.average())


def evaluate_input_split(model, loader, epoch):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()
    all_time_meter = AverageMeter()
    data_time = AverageMeter()

    acc_meter_30m = AverageMeter()
    intersection_meter_30m = AverageMeter()
    union_meter_30m = AverageMeter()

    acc_meter_60m = AverageMeter()
    intersection_meter_60m = AverageMeter()
    union_meter_60m = AverageMeter()

    acc_meter_90m = AverageMeter()
    intersection_meter_90m = AverageMeter()
    union_meter_90m = AverageMeter()

    acc_meter_120m = AverageMeter()
    intersection_meter_120m = AverageMeter()
    union_meter_120m = AverageMeter()

    acc_meter_per_height = {30: acc_meter_30m, 60: acc_meter_60m, 90: acc_meter_90m, 120: acc_meter_120m}
    intersection_meter_per_height = {30: intersection_meter_30m, 60: intersection_meter_60m, 90: intersection_meter_90m,
                                     120: intersection_meter_120m}
    union_meter_per_height = {30: union_meter_30m, 60: union_meter_60m, 90: union_meter_90m, 120: union_meter_120m}

    from _collections import defaultdict
    num_per_height = defaultdict(int)

    # colors, class_names = parse_csv(cfg)
    colors, class_names = parse_csv_merged(cfg)

    model.eval()
    save_path = epoch
    logger.info('[Epoch {} Eval Summary]:'.format(epoch))
    # pbar = tqdm(total=len(loader))
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # if i > 100:
        #     break
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_split_list = batch_data['img_data']
        torch.cuda.synchronize()
        #         tic = time.perf_counter()
        #
        # image_name = batch_data['info'].split('/')[-2]
        # index_1 = image_name.find(')')
        # height = int(image_name[index_1 + 1:])
        # num_per_height[height] += 1
        ### extract height
        height = extract_height(os.path.dirname(batch_data['info']))

        data_time.update(time.perf_counter() - tic)

        with torch.no_grad():
            # print(batch_data['img_ori'].shape)
            # segSize = (int(batch_data['img_ori'].shape[0]),
            #            int(batch_data['img_ori'].shape[1]))
            segSize = (seg_label.shape[0], seg_label.shape[1])
            # print("seg", segSize)
            pred = torch.zeros(segSize[0], segSize[1]).cuda()
            import math
            # print(len(img_resized_split_list))
            split_num = int(math.sqrt(len(img_resized_split_list)))
            # import pdb; pdb.set_trace()
            # print(split_num)
            img_split_coord_list = [(i, j) for i in range(split_num) for j in range(split_num)]
            split_height = int(segSize[0] / split_num)
            split_width = int(segSize[1] / split_num)
            for split_id in range(len(img_resized_split_list)):
                img_split_coord = img_split_coord_list[split_id]
                up_h = img_split_coord[0] * split_height
                left_w = img_split_coord[1] * split_width

                scores = torch.zeros(1, cfg.DATASET.num_class, split_height, split_width).cuda()
                # img_split_ori = batch_data['img_ori'][split_id]
                img_resized_list = img_resized_split_list[split_id]

                torch.cuda.synchronize()
                start = time.time()
                for img in img_resized_list:
                    img = img.cuda()
                    # print(img.size())

                    if cfg.MODEL.arch == "hrnet_ocr_v1":
                        # import pdb; pdb.set_trace()
                        if cfg.MODEL.use_teacher:
                            scores_tmp = model(img, batch_data['seg_label'], True, up_h, up_h + split_height, left_w, left_w + split_width)[0]
                        else:

                            scores_tmp = model(img, batch_data['seg_label'])[0]  # <class 'tuple'> 2
                            # <class 'tuple'> 2

                    elif cfg.MODEL.arch == "hrnet_ocr.HRNet_Mscale":
                        inputs = {'images': img, 'gts': seg_label}
                        scores_tmp = model(inputs)

                    else:
                        scores_tmp = model(img)

                    # print(type(scores_tmp), scores_tmp.size())

                    scores_tmp = nn.functional.interpolate(
                        scores_tmp, size=(split_height, split_width), mode='bilinear', align_corners=False)
                    scores_tmp = nn.functional.softmax(scores_tmp, dim=1)
                    scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                #                 scores[:, -1, :, :] = 0  # ignore the last category

                torch.cuda.synchronize()
                end = time.time()
                print("spiltA inference time:", end - start)
                exit()

                _, pred_tmp = torch.max(scores, dim=1)

                pred[up_h:up_h + split_height, left_w:left_w + split_width] = pred_tmp.squeeze(0)
            pred = as_numpy(pred.cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        # print(pred.shape, seg_label.shape)
        class_list = range(cfg.DATASET.num_class)
        #         print(class_list[-2:])
        acc, pix = accuracy(pred, seg_label, ignored_label=class_list[-2:])
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        # print(intersection, union)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        if height != -1:
            acc_meter_per_height[height].update(acc, pix)
            intersection_meter_per_height[height].update(intersection)
            union_meter_per_height[height].update(union)

        # logger.info()
        # visualization
        if cfg.VAL.visualize:
            if not os.path.isdir(os.path.join(cfg.DIR,
                                              '{}_split_valresult_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
                                                                                           cfg.DATASET.imgValMaxSize,
                                                                                           str(save_path),
                                                                                           height))):
                os.makedirs(os.path.join(cfg.DIR,
                                         '{}_split_valresult_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
                                                                                      cfg.DATASET.imgValMaxSize,
                                                                                      str(save_path), height)))

            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR,
                             '{}_split_valresult_valsize{}/{}/{}m'.format(cfg.DATASET.val_data_name,
                                                                          cfg.DATASET.imgValMaxSize,
                                                                          str(save_path), height)),
                colors
            )

        if cfg.VAL.save_labelfile:
            if not os.path.isdir(os.path.join(cfg.DIR,
                                              '{}_split_pred_labelfile_valsize{}'.format(cfg.DATASET.val_data_name,
                                                                                         cfg.DATASET.imgValMaxSize),
                                              str(save_path))):
                os.makedirs(os.path.join(cfg.DIR, '{}_split_pred_labelfile_valsize{}'.format(cfg.DATASET.val_data_name,
                                                                                             cfg.DATASET.imgValMaxSize),
                                         str(save_path)))

            save_labelfile(
                batch_data['info'],
                pred,
                os.path.join(cfg.DIR, '{}_split_pred_labelfile_valsize{}'.format(cfg.DATASET.val_data_name,
                                                                                 cfg.DATASET.imgValMaxSize),
                             str(save_path))
            )

        # pbar.update(1)
        # pbar.close()
        # summary

        iou = intersection_meter.sum / (union_meter.sum + 1e-10)

        all_time_meter.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        logger.info(
            'Rank[{}] Epoch: [{}][{}/{}] Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Data Time: {:.4f}s, Inference Time: {:.4f}s, Total Time: {:.4f}s'
                .format(dist.get_rank(), epoch, i, len(loader), iou[:-2].mean(), iou.mean(), acc_meter.average() * 100,
                        data_time.average(), time_meter.average(), all_time_meter.average()))
    # torch.cuda.synchronize()

    for i, _iou in enumerate(iou):
        logger.info('Rank[{}] class [{}] {}, IoU: {:.4f}'.format(dist.get_rank(), i, class_names[i], _iou))

        # save_dict[class_names[i]] = _iou

    logger.info(
        'Final Result for split validation: Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Data Time: {:.4f}s, Inference Time: {:.4f}s, Total Time: {:.4f}s'
            .format(iou[:-2].mean(), iou.mean(), acc_meter.average() * 100, data_time.average(), time_meter.average(),
                    all_time_meter.average()))

    save_dict = defaultdict(dict)
    save_dict['all']['inter'] = intersection_meter.sum
    save_dict['all']['uni'] = union_meter.sum
    # print(acc_meter.average(), acc_meter.sum, len(loader), acc_meter.count)
    # print(time_meter.average(), time_meter.sum, time_meter.count)
    save_dict['all']["pixel_acc"] = acc_meter.sum * 100
    # save_dict['all']["average_time"] = time_meter.sum
    save_dict['all']["num"] = len(loader)
    save_dict['all']["pixel_count"] = acc_meter.count

    for h in [30, 60, 90, 120]:
        save_dict[h]['inter'] = intersection_meter_per_height[h].sum
        save_dict[h]['uni'] = union_meter_per_height[h].sum
        # print(acc_meter.average(), acc_meter.sum, len(loader), acc_meter.count)
        # print(time_meter.average(), time_meter.sum, time_meter.count)
        save_dict[h]["pixel_acc"] = acc_meter_per_height[h].sum * 100
        # save_dict[h]["average_time"] = time_meter.sum
        save_dict[h]["num"] = num_per_height[h]
        save_dict[h]["pixel_count"] = acc_meter_per_height[h].count

    path_name = os.path.join(cfg.DIR, 'dist_split_result_size{}'.format(cfg.DATASET.imgValMaxSize), str(save_path))
    if not os.path.isdir(path_name):
        os.makedirs(path_name)
    with open(os.path.join(path_name, 'dist_rank{}.pkl'.format(dist.get_rank())), 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    torch.cuda.synchronize()
    iou_mean.append(iou.mean())
    iou_mean_except_ignore.append(iou[:-2].mean())
    acc_mean.append(acc_meter.average() * 100)
    inference_mean.append(time_meter.average())



def main(epoch, model, configer, use_split=False):
    # Dataset and Loader
    import torch.distributed as dist
    gpu_nums = len(configer.get('gpu'))
    batch_size = 1

    if use_split:
        dataset_val = ValDataset_Split_cv(
            configer.get('xiashi', 'root_dataset'),
            configer.get('xiashi', 'list_val'),
            configer,
            configer.get('xiashi', 'crop_type'),
            cfg.VAL.val_height,
            world_size=gpu_nums, rank=dist.get_rank())

    else:
        dataset_val = ValDataset_cv(
            configer.get('xiashi', 'root_dataset'),
            configer.get('xiashi', 'list_val'),
            configer,
            world_size=gpu_nums, rank=dist.get_rank())

    loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=configer.get('data', 'workers') // gpu_nums,
            drop_last=True)

    # logger.info("rank {}".format(dist.get_rank()))
    # print(dist.get_rank(), len(dataset_val), len(loader_val))

    model.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank, ],
            output_device=local_rank
        )

    # Main loop
    if cfg.VAL.use_split:
        evaluate_input_split(model, loader_val, epoch)
    else:
        evaluate(model, loader_val, epoch)
    # evaluate(model, loader_val, cfg, gpu, save_path)

    logger.info('Evaluation Done!')




def val_xiashidata(model, use_split=False, configer):
    import torch.multiprocessing as mp
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    args = parse_args()
    assert_and_infer_cfg(args)
    cfg_from_file(args.cfg, args)

    # # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    rank2gpu = {i: gpus[i] for i in range(num_gpus)}

    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:{}'.format(args.port),
                                world_size=num_gpus,
                                rank=args.local_rank
                                )

    epochs = parseIntSet(cfg.VAL.multi_checkpoints)
    epochs = [sorted(epochs)[-1]]
    import datetime

    ISOTIMEFORMAT = '%Y-%m-%d-%H:%M:%S'
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
    if "ocr" in cfg.MODEL.arch and cfg.OCR.if_object_mask:
        cfg.DIR = cfg.DIR + '_objectmask'
    if cfg.TRAIN.use_ohem:
        cfg.DIR = cfg.DIR + '_ohem'
    if cfg.TRAIN.se_loss:
        cfg.DIR = cfg.DIR + '_se'
    if cfg.TRAIN.metric_loss:
        cfg.DIR = cfg.DIR + '_metric'

    if len(cfg.VAL.checkpoint) == 0:
        # datetime.datetime.now().strftime(ISOTIMEFORMAT)
        if cfg.VAL.use_split:
            log_filename = '{}'.format(cfg.DATASET.val_data_name) + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) + \
                           '_dist_val_split_' + 'epoch{}_{}'.format(epochs[0], epochs[-1]) + '.txt'
        else:
            log_filename = '{}'.format(cfg.DATASET.val_data_name) + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) + \
                           '_dist_val_' + 'epoch{}_{}'.format(epochs[0], epochs[-1]) + '.txt'
        logger = setup_logger(name='Logger', distributed_rank=dist.get_rank(), save_dir=cfg.DIR,
                              filename=log_filename,
                              mode='w')
        logger.info("Saving log to {}/{}".format(cfg.DIR, log_filename))
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))

        for epoch in epochs:
            cfg.MODEL.weights = os.path.join(
                cfg.DIR, '{}_epoch_{}.pth'.format(cfg.MODEL.arch, str(epoch)))

            logger.info("load weights path: {}".format(cfg.MODEL.weights))

            assert os.path.exists(cfg.MODEL.weights), "checkpoint does not exist!"

            if not os.path.isdir(os.path.join(cfg.DIR)):
                os.makedirs(os.path.join(cfg.DIR))

            main(epoch, model, configer, use_split=use_split)

        logger.info(
            "######################################### validation summary #########################################")
        if cfg.VAL.use_split:
            for i, epoch in enumerate(epochs):
                logger.info(
                    'Split val results for Ckpt_{}, Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
                        .format(epoch, iou_mean_except_ignore[i], iou_mean[i], acc_mean[i], inference_mean[i]))
        else:
            for i, epoch in enumerate(epochs):
                logger.info(
                    'Ckpt_{}, Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
                        .format(epoch, iou_mean_except_ignore[i], iou_mean[i], acc_mean[i], inference_mean[i]))
    else:
        # datetime.datetime.now().strftime(ISOTIMEFORMAT)
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

        if cfg.VAL.use_split:
            log_filename = '{}'.format(cfg.DATASET.val_data_name) + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) + \
                           '_dist_val_split' + '.txt'
        else:
            log_filename = '{}'.format(cfg.DATASET.val_data_name) + '_valsize{}'.format(cfg.DATASET.imgValMaxSize) + \
                           '_dist_val' + '.txt'
        logger = setup_logger(name='Logger', distributed_rank=dist.get_rank(), save_dir=cfg.DIR,
                              filename=log_filename,
                              mode='w')
        logger.info("Saving log to {}/{}".format(cfg.DIR, log_filename))
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))

        cfg.MODEL.weights = cfg.VAL.checkpoint
        logger.info("load weights path: {}".format(cfg.MODEL.weights))
        assert os.path.exists(cfg.MODEL.weights), "checkpoint does not exist!"

        if not os.path.isdir(os.path.join(cfg.DIR)):
            os.makedirs(os.path.join(cfg.DIR))
        main(os.path.basename(cfg.VAL.checkpoint).replace('.pth', ''))

        logger.info(
            "######################################### validation summary #########################################")
        if cfg.VAL.use_split:
            logger.info(
                'Split val results for saved weights: Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
                    .format(iou_mean_except_ignore[0], iou_mean[0], acc_mean[0], inference_mean[0]))
        else:
            logger.info(
                'saved weights: Mean IOU (except_ignore & people): {:.4f}, Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
                    .format(iou_mean_except_ignore[0], iou_mean[0], acc_mean[0], inference_mean[0]))

        logger.info("Saving log to {}/{}".format(cfg.DIR, log_filename))