import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import torch.nn as nn

def extract_height(str1):
    height_list = [30, 60, 90, 120]
    numbers = list(map(lambda y:str(y), range(10)))

    height_list = list(map(lambda y:str(y), height_list))
    lenth1=len(str1)

    extracted_result = []
    for height in height_list:
        lenth2 = len(height)
        index_list = indexstr(str1, height)
        # print(index_list)
        index_result = []
        for index in index_list:
            if str1[index - 1] == '/': # at the beginning of the name
                if (str1[index + lenth2] == "m") or (str1[index + lenth2] == "mi"):
                    index_result.append(index)
            elif (str1[index - 1] not in numbers) and (
                    str1[index - 1] != '.') and (str1[index - 1] != ':') and (str1[index - 1] != '/'):  # the character before this word should not be a number
                if index + lenth2 < lenth1:
                    if (str1[index + lenth2] not in numbers): # the character after this word should not be a number
                        index_result.append(index)
                else:
                    index_result.append(index)
        if len(index_result) != 0:
            if len(index_result) == 1:
                extracted_result.append(str1[index_result[0]: index_result[0]+lenth2])
            else: # maybe because the same name appears twice in the path
                extracted_result.append(str1[index_result[0]: index_result[0] + lenth2])
                # print(str1)
                # for each in range(len(index_result)):
                #     print(str1[index_result[each]: index_result[each]+lenth2])
    if (len(extracted_result) == 0) or (len(extracted_result)) > 1 :
        return -1
    else:
        return int(extracted_result[0])





def indexstr(str1,str2):
    '''查找指定字符串str1包含指定子字符串str2的全部位置，
    以列表形式返回'''

    lenth2=len(str2)
    lenth1=len(str1)
    indexstr2=[]
    i=0
    while str2 in str1[i:]:
        indextmp = str1.index(str2, i, lenth1)
        indexstr2.append(indextmp)
        i = (indextmp + lenth2)
    return indexstr2

def my_load_state_dict(model, state_dict, isteacher=False):
    state_dict = state_dict['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if isteacher:
        for k, v in state_dict.items():
            new_state_dict[k] = v
    else:
        for k, v in state_dict.items():
            name = k[7:]            # remove `module.
            new_state_dict[name] = v

    state_dict = new_state_dict
    own_state_old = model.state_dict()
    print(list(own_state_old.keys())[:5])
    # print("\n\n\n");
    print(list(new_state_dict.keys())[:5])
    print("own model params # %d" % len(own_state_old.keys()))
    # for name, param in own_state_old.items():
    #     if name not in state_dict:
    #         print(name)
    count_used = 0
    count_unused = 0
    count_all = 0
    for name, param in state_dict.items():
        count_all += 1
        if name not in own_state_old:
            # print(name)
            count_unused += 1
            continue
        if param.size() != own_state_old[name].size():
            print(name)
            count_unused += 1
            continue
        count_used += 1
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state_old[name].copy_(param)
    print("finetune model params #%d/%d" % (count_used, count_all))
    print("Unused model params #%d/%d" % (count_unused, count_all))

# def setup_logger(distributed_rank=0, filename="log.txt"):
#     logger = logging.getLogger("Logger")
#     logger.setLevel(logging.DEBUG)
#     # don't log results for the non-master process
#     if distributed_rank > 0:
#         return logger
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
#     ch.setFormatter(logging.Formatter(fmt))
#     logger.addHandler(ch)
#
#     return logger
def setup_logger(name, distributed_rank, save_dir=None, filename="log.txt", mode='w'):
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


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / (self.count + 1e-10)

    def value(self):
        return self.val

    def average(self):
        return self.avg



def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')

    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
        # labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
        #                 np.tile(colors[label + 1],
        #                         (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label, ignored_label=[-1]):
    # print(label, label.shape, label>=0, label != 16)
    valid = (label >= 0)
    for each in ignored_label:
        valid = valid * (label != each)
    # valid = (label >= 0) * (label != ignored_label)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    # imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret

class UnNormalize(object):
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

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

if __name__ == "__main__":
    # print(parseIntSet("<20"))

    string = '/workdir/hrr/data/xiashi_data/segdata_8/process_gt_v1.4/8/wende/annotations/tianjin/20200323/3.20-xiawu30mi/3.20-13:28_jintanggongluzhaopian1/zhaopian/label_map_merged_v1.4/DJI_0778.png'
    height = extract_height(os.path.dirname(string))
    print(height)

    string = '/workdir/hrr/data/xiashi_data/segdata_8/process_gt_v1.4/8/wende/annotations/tianjin/20200323/3.30-xiawu30mi/3.30-13:28_jintanggongluzhaopian1/zhaopian/label_map_merged_v1.4/DJI_0778.png'
    height = extract_height(os.path.dirname(string))
    print(height)

    string = "/workdir/hrr/data/xiashi_data/segdata_5/data_pinyin/5/beisai/images/0605shangwu1gongmingnanhuandadao(daoluyanxian)90/900605010007.JPG"
    height = extract_height(os.path.dirname(string))
    print(height)

    string = "/workdir/hrr/data/xiashi_data/segdata_7/data_pinyin/7/beisai/images/tianjin/meituan3yuedisizhoushuju/3.24/3.24-15:10(quyu8)60/DJI_0530.JPG"
    height = extract_height(os.path.dirname(string))
    print(height)

    string = "/workdir/hrr/data/xiashi_data/segdata_7/data_pinyin/7/beisai/images/tianjin/meituan3yuedisizhoushuju/3.25/3.25-15:30(zhongyangdadao)90/dierjiaci/zhaopian/DJI_0745.JPG"
    height = extract_height(os.path.dirname(string))
    print(height)
