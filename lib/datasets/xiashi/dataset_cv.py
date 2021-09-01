# from .dataset import BaseDataset, imresize
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch, os
import torchvision.transforms.functional as TF
from torchvision import transforms
import random, math
import cv2
import albumentations as A
import albumentations.augmentations.functional as F
import json
import time
from lib.datasets.xiashi.utils import extract_height

classnum = torch.zeros(4, 20)


def user_collate_fn(batch):
    assert (len(batch) == 1)
    #     print("batch", batch[0]['img_data'].size())
    return batch[0]


def user_scattered_collate(batch):
    return batch


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset_cv(torch.utils.data.Dataset):
    def __init__(self, odgt, configer, **kwargs):
        # parse options
        self.imgSizes = configer.get('xiashi', 'imgSizes')
        self.imgMaxSize = configer.get('xiashi', 'imgMaxSize')

        self.imgValSizes = configer.get('xiashi', 'imgValSizes')
        self.imgValMaxSize = configer.get('xiashi', 'imgValMaxSize')

        self.imgTestSizes = configer.get('xiashi', 'imgTestSizes')
        self.imgTestMaxSize = configer.get('xiashi', 'imgTestMaxSize')

        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = configer.get('xiashi', 'padding_constant')

        # parse the input list
        self.parse_input_list(odgt, **kwargs)
        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # self.countnum = 0
        # self.toTensor = ToTensorV2()

    def parse_input_list(self, odgt, world_size=1, rank=0, start_idx=-1, end_idx=-1):

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):  # here
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
            # print("listsample:", self.list_sample[0])
            # for index,x in enumerate(open(odgt, 'r')):
            #     if index == 0:
            #         print(x)
            # print("odgt:", odgt)
        num_total = len(self.list_sample)
        # exit()

        if world_size > 1:
            self.num_sample = int(math.ceil(num_total * 1.0 / world_size))
            self.start_idx = rank * self.num_sample
            self.end_idx = min(self.start_idx + self.num_sample, num_total)
        else:
            self.start_idx = 0
            self.end_idx = num_total
            self.num_sample = num_total

        # assert self.num_sample > 0
        print('Dataset Samples #total: {}, #process [{}]: {}-{}'
              .format(num_total, rank, self.start_idx, self.end_idx))

    def img_transform(self, img):
        # 0-255 to 0-1
        # img = np.float32(np.array(img)) / 255.
        # print(img.shape, img[0,0,:], np.float32(img).shape, np.float32(img)[0,0,:])
        img = np.float32(img) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        # img = self.normalize(image=img)
        # img = img['image']
        # img = self.toTensor(image=img)
        # img = img['image']
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        # segm = torch.from_numpy(np.array(segm)).long() - 1
        segm = torch.from_numpy(segm).long()  # use unlabel classes
        # segm = self.toTensor(image=segm)
        # segm = segm['image'].long()
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset_cv(BaseDataset_cv):
    def __init__(self, root_dataset, odgt, configer, crop_type, batch_per_gpu=1, **kwargs):
        super(TrainDataset_cv, self).__init__(odgt, configer, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm label
        self.segm_downsampling_rate = 1
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        # self.cur_idx = 0
        # self.if_shuffled = False
        self.cur_idx = self.start_idx
        self.crop_type = crop_type

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.list_sample)

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]

            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.end_idx:
                self.cur_idx = self.start_idx
            # if self.cur_idx >= self.num_sample:
            #     self.cur_idx = 0
            #     np.random.shuffle(self.list_sample) # shuffle if to the end, the next epoch

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def _largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
        Converted to Python by Aaron Snoswell
        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def transform(self, image, segm, this_record):

        rgb_trans = A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)
        bright_trans = A.RandomBrightnessContrast(p=0.5)
        hue_trans = A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, always_apply=False,
                                         p=0.5)
        blur_trans = A.augmentations.transforms.GaussianBlur(p=0.2)
        shadow_trans = A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5,
                                      shadow_roi=(0, 0.5, 1, 1), p=1)

        image_trans = A.Compose(
            [
                # rgb_trans,
                # bright_trans,
                hue_trans,
                # blur_trans,
                blur_trans,  # lxy
                shadow_trans,
                # A.RandomContrast(limit=0.2, always_apply=False, p=0.5)
            ])
        image = image_trans(image=image)
        image = image['image']

        # affine_trans = A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=90, p=0.5)
        affine_trans = A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5)
        hflip_trans = A.HorizontalFlip(p=0.5)
        vfip_trans = A.VerticalFlip(p=0.5)
        joint_trans = A.Compose(
            [
                affine_trans,
                hflip_trans,
                vfip_trans,
                # A.RandomCrop(32, 32, always_apply=False, p=1.0),
                # A.RandomRotate90(always_apply=False, p=0.5)
            ])
        augmented = joint_trans(image=image, mask=segm)
        image, segm = augmented['image'], augmented['mask']

        return image, segm

    def get_crop_params(self, image, output_size):
        # w, h = _get_image_size(img)
        h, w = image.shape[0:2]
        # print("h", h, w)
        th, tw = output_size
        # print("th", th, tw)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        # print("i", i)
        j = random.randint(0, w - tw)
        # print("j", j)
        h1 = i
        w1 = j
        h2 = i + th
        w2 = j + tw
        # print("h1", h1, w1, h2, w2)
        # h1, w1, h2, w2 = i, j, i+th, j+tw
        return h1, w1, h2, w2

    def crop_transform(self, image, segm, this_record, crop_type="crop_0"):

        height = extract_height(os.path.dirname(this_record['fpath_img']))

        global classnum
        # #cal
        # count = 0
        # for j in np.unique(segm):
        #     if height == 30:
        #         classnum[0][j] += 1
        #         count += 1
        #     elif height == 60:
        #         classnum[1][j] += 1
        #         count += 1
        #     elif height == 90:
        #         classnum[2][j] += 1
        #         count += 1
        #     elif height == 120:
        #         classnum[3][j] += 1
        #         count += 1
        # print("\nadd:", count)
        # print("\n num:", classnum)

        if height != -1:
            crop_type = crop_type
        else:
            crop_type = "crop_0"

        # print(crop_type)
        if crop_type == "crop_0":
            if random.random() > 0.2:
                w_orig, h_orig = this_record['width'], this_record['height']
                crop_size = (h_orig - random.randint(0, 1000), w_orig - random.randint(0, 1000))
                # print(crop_size)
                # Random crop
                h1, w1, h2, w2 = self.get_crop_params(
                    image, output_size=crop_size)
                image = image[h1:h2, w1:w2]
                segm = segm[h1:h2, w1:w2]

        if crop_type == "input_split":
            if random.random() > 0.2:
                w_orig, h_orig = this_record['width'], this_record['height']
                if height <= 60:
                    crop_size = (h_orig - random.randint(0, 1000), w_orig - random.randint(0, 1000))
                else:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 5.0)), w_orig - random.randint(0, int(w_orig / 5.0)))

                # Random crop
                h1, w1, h2, w2 = self.get_crop_params(
                    image, output_size=crop_size)
                image = image[h1:h2, w1:w2]
                segm = segm[h1:h2, w1:w2]

        if crop_type == "crop_A":
            # print(height)
            if height < 60:
                if random.random() > 0.2:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 8.0)), w_orig - random.randint(0, int(h_orig / 8.0)))
                    # print("orih:",h_orig, "oriw:", w_orig, height, crop_size)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

            elif height == 60:
                if random.random() > 0.2:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 4.0)), w_orig - random.randint(0, int(h_orig / 4.0)))
                    # print("orih:",h_orig, "oriw:", w_orig, height, crop_size)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

            else:
                tmp = random.random()
                if tmp > 0.2 and tmp < 0.4:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    # crop_size = (h_orig - random.randint(1000, int(h_orig/2.0)), w_orig - random.randint(1000, int(w_orig/2.0)))
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 3.0)), w_orig - random.randint(0, int(h_orig / 3.0)))
                    # print("orih:",h_orig, "oriw:", w_orig, height, "1", crop_size)

                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

                else:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    crop_size = (h_orig - random.randint(int(h_orig / 3.0), int(h_orig / 2.0)),
                                 w_orig - random.randint(int(h_orig / 3.0), int(w_orig / 2.0)))
                    # print(height, "2", crop_size)
                    # crop_size = (h_orig - random.randint(0, 1000), w_orig - random.randint(0, 1000))

                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
                    # image.save('tmp_2_' + os.path.basename(this_record['fpath_img']))

        if crop_type == "crop_B":
            # print(height)
            w_orig, h_orig = this_record['width'], this_record['height']
            if height < 60:
                if random.random() > 0.3:

                    this_scale = min(
                        2000 / min(h_orig, w_orig), \
                        2500 / max(h_orig, w_orig))
                    w_crop = int(w_orig * this_scale)
                    h_crop = int(h_orig * this_scale)

                    # crop_size = (
                    # h_orig - random.randint(0, int(h_orig / 8.0)), w_orig - random.randint(0, int(h_orig / 8.0)))
                    crop_size = (h_crop, w_crop)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
                else:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 8.0)), w_orig - random.randint(0, int(h_orig / 8.0)))
                    # crop_size = (2000, 2000)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

            elif height == 60:
                if random.random() > 0.3:
                    # crop_size = (
                    # h_orig - random.randint(0, int(h_orig / 4.0)), w_orig - random.randint(0, int(h_orig / 4.0)))
                    this_scale = min(
                        2000 / min(h_orig, w_orig), \
                        2500 / max(h_orig, w_orig))
                    w_crop = int(w_orig * this_scale)
                    h_crop = int(h_orig * this_scale)
                    crop_size = (h_crop, w_crop)
                    # print(crop_size)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
                else:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 4.0)), w_orig - random.randint(0, int(h_orig / 4.0)))
                    # crop_size = (2000, 2000)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
            else:
                tmp = random.random()
                if tmp > 0.3:
                    # crop_size = (
                    # h_orig - random.randint(0, int(h_orig / 3.0)), w_orig - random.randint(0, int(h_orig / 3.0)))
                    # crop_size = (1000, 1000)
                    this_scale = min(
                        1000 / min(h_orig, w_orig), \
                        1500 / max(h_orig, w_orig))
                    w_crop = int(w_orig * this_scale)
                    h_crop = int(h_orig * this_scale)
                    crop_size = (h_crop, w_crop)
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
                else:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 3.0)), w_orig - random.randint(0, int(h_orig / 3.0)))
                    # crop_size = (1000, 1000)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

        if crop_type == "crop_C":
            # print(crop_type)
            # print(height)
            w_orig, h_orig = this_record['width'], this_record['height']
            if height < 60:
                if random.random() > 0.2:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 8.0)), w_orig - random.randint(0, int(h_orig / 8.0)))
                    # crop_size = (2000, 2000)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

            elif height == 60:
                if random.random() > 0.3:
                    # crop_size = (
                    # h_orig - random.randint(0, int(h_orig / 4.0)), w_orig - random.randint(0, int(h_orig / 4.0)))
                    this_scale = min(
                        2000 / min(h_orig, w_orig), \
                        2500 / max(h_orig, w_orig))
                    w_crop = int(w_orig * this_scale)
                    h_crop = int(h_orig * this_scale)
                    crop_size = (h_crop, w_crop)
                    # print(crop_size)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
                else:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 4.0)), w_orig - random.randint(0, int(h_orig / 4.0)))
                    # crop_size = (2000, 2000)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
            else:
                tmp = random.random()
                if tmp > 0.3:
                    # crop_size = (
                    # h_orig - random.randint(0, int(h_orig / 3.0)), w_orig - random.randint(0, int(h_orig / 3.0)))
                    # crop_size = (1000, 1000)
                    this_scale = min(
                        1000 / min(h_orig, w_orig), \
                        1500 / max(h_orig, w_orig))
                    w_crop = int(w_orig * this_scale)
                    h_crop = int(h_orig * this_scale)
                    crop_size = (h_crop, w_crop)
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
                else:
                    crop_size = (
                        h_orig - random.randint(0, int(h_orig / 3.0)), w_orig - random.randint(0, int(h_orig / 3.0)))
                    # crop_size = (1000, 1000)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]

        if crop_type == "crop_L":
            # print(height)
            if height < 60:
                if random.random() > 0.2:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    crop_size = (int(h_orig / 8.0)), int(w_orig / 8.0)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

            elif height == 60:
                if random.random() > 0.2:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    crop_size = (int(h_orig / 8.0)), int(w_orig / 8.0)
                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]
            else:
                tmp = random.random()
                if tmp > 0.2 and tmp < 0.4:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    # crop_size = (h_orig - random.randint(1000, int(h_orig/2.0)), w_orig - random.randint(1000, int(w_orig/2.0)))
                    crop_size = (int(h_orig / 8.0)), int(w_orig / 8.0)
                    # print(height, "1", crop_size)

                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

                else:
                    w_orig, h_orig = this_record['width'], this_record['height']
                    crop_size = (int(h_orig / 8.0)), int(w_orig / 8.0)
                    # print(height, "2", crop_size)
                    # crop_size = (h_orig - random.randint(0, 1000), w_orig - random.randint(0, 1000))

                    # Random crop
                    h1, w1, h2, w2 = self.get_crop_params(
                        image, output_size=crop_size)
                    image = image[h1:h2, w1:w2]
                    segm = segm[h1:h2, w1:w2]

        return image, segm

    def __getitem__(self, index):
        # get sub-batch candidates
        import time
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes
        # print("short", this_short_size)
        # this_short_size = this_short_size * 2
        # self.imgMaxSize = self.imgMaxSize * 2
        # print("short", this_short_size)

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        # print(batch_widths, batch_heights, batch_width, batch_height)
        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = (-1 * torch.ones(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate)).long()

        # print("prepare", time.time() - tic)

        # print(batch_segms.size(), batch_height // self.segm_downsampling_rate,  self.segm_downsampling_rate) # 1
        for i in range(self.batch_per_gpu):
            # print(batch_width, batch_widths[i], batch_height, batch_heights[i])
            import time
            this_record = batch_records[i]

            # load image and label
            image_path = this_record['fpath_img']
            segm_path = this_record['fpath_segm']

            tic = time.time()
            img = cv2.imread(image_path)
            segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
            ### convert for augmentation
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # assert(segm.mode == "L")
            assert (img.shape[0] == segm.shape[0])
            assert (img.shape[1] == segm.shape[1])
            #             print(img.shape, segm.shape)

            tic = time.time()
            img, segm = self.crop_transform(img, segm, this_record, self.crop_type)

            img = F.resize(img, batch_heights[i], batch_widths[i], interpolation=cv2.INTER_LINEAR)
            segm = F.resize(segm, batch_heights[i], batch_widths[i], interpolation=cv2.INTER_NEAREST)

            img, segm = self.transform(img, segm, this_record)

            img = self.img_transform(img)
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img  # the locations outside the img pixels are zero
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10)  # It's a fake length due to the trick that every loader maintains its own list


class ValDataset_cv(BaseDataset_cv):
    def __init__(self, root_dataset, odgt, configer, **kwargs):
        super(ValDataset_cv, self).__init__(odgt, configer, **kwargs)
        self.root_dataset = root_dataset
        self.list_sample = self.list_sample[self.start_idx: self.end_idx]

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label

        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

        img = cv2.imread(image_path)
        segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
        ### convert for augmentation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # h,w,c

        # img = Image.open(image_path).convert('RGB')
        # segm = Image.open(segm_path)
        assert (img.shape[0] == segm.shape[0])
        assert (img.shape[1] == segm.shape[1])

        # ori_width, ori_height = img.size
        # print(img.shape)
        ori_height, ori_width = img.shape[:-1]      # 3648, 4864


        img_resized_list = []
        for this_short_size in self.imgValSizes:
            # calculate target height and width

            scale = min(this_short_size / float(min(ori_height, ori_width)),    # ori:(3648, 4864)   scale:0.2741
                        self.imgValMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)  # 1824, 2432

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant) # 1880
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)  # 2504

            # resize images
            # img_resized = imresize(img, (target_width, target_height), interp='bilinear')
            img_resized = F.resize(img, target_height, target_width, interpolation=cv2.INTER_LINEAR)

            # print("imgdata size:", target_height, target_width) # (1880, 2504, 3)

            # print(img_resized.shape)
            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # avoid output original size, out of memory
        segm = F.resize(segm, int(ori_height / 2), int(ori_width / 2), interpolation=cv2.INTER_NEAREST) # (1824, 2432)
        # segm = F.resize(segm, target_height, target_width, interpolation=cv2.INTER_NEAREST)  # (1880, 2504)
        # print("seg_label size:", segm.shape)

        # segm = imresize(segm, (int(ori_width / 2), int(ori_height / 2)), interp='nearest')
        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)  # (3648, 4864, 3)
        output['img_data'] = [x.contiguous() for x in
                              img_resized_list]  # [1, 3, 528, 704], [1, 3, 600, 800], [1, 3, 752, 1000]
        output['seg_label'] = batch_segms.contiguous()  # [1, 1824, 2432]
        #         print(output['seg_label'])
        output['info'] = this_record['fpath_img']

        # print("imgdata size:", output['img_data'][0].size()) # ([1, 3, 1880, 2504])
        # print("seg_label size:", output['seg_label'].size()) # ([1, 1824, 2432])
        return output

    def __len__(self):
        # return self.num_sample
        return len(self.list_sample)


class ValDataset_Split_cv(BaseDataset_cv):
    def __init__(self, root_dataset, odgt, crop_type, val_height, **kwargs):
        super(ValDataset_Split_cv, self).__init__(odgt, **kwargs)
        self.root_dataset = root_dataset
        # print(self.start_idx, self.end_idx)
        self.list_sample = self.list_sample[self.start_idx: self.end_idx]
        # print(len(self.list_sample))
        self.crop_type = crop_type
        self.val_height = val_height

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = cv2.imread(image_path)
        segm = cv2.imread(segm_path, cv2.IMREAD_UNCHANGED)
        ### convert for augmentation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # h,w,c

        assert (img.shape[0] == segm.shape[0])
        assert (img.shape[1] == segm.shape[1])

        # ori_width, ori_height = img.size
        # print(img.shape)
        ori_height, ori_width = img.shape[:-1]

        if self.val_height == -1:
            height = extract_height(os.path.dirname(this_record['fpath_img']))
        else:
            height = self.val_height

        if self.crop_type == "crop_A":
            split_num = int(math.ceil(float(height) / 60))
        if self.crop_type == "crop_B":
            split_num = int(math.ceil(float(height) / 60) * 2)
        if self.crop_type == "crop_C":
            if height == 30:
                split_num = 1
            else:
                split_num = int(math.ceil(float(height) / 60) * 2)
        #         print(height, split_num)

        img_split_coord_list = [(i, j) for i in range(split_num) for j in range(split_num)]

        img_ori_split_list = []
        img_resized_split_list = []
        for split_id in range(split_num ** 2):
            img_split_coord = img_split_coord_list[split_id]
            up_h = img_split_coord[0] * int(ori_height / split_num)
            left_w = img_split_coord[1] * int(ori_width / split_num)

            img_split = img[up_h:up_h + int(ori_height / split_num), left_w:left_w + int(ori_width / split_num)]

            img_ori_split_list.append(np.array(img_split))

            img_resized_list = []
            for this_short_size in self.imgValSizes:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            self.imgValMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)

                # to avoid rounding in network
                target_width = self.round2nearest_multiple(target_width, self.padding_constant)
                target_height = self.round2nearest_multiple(target_height, self.padding_constant)

                # resize images
                # img_resized = imresize(img_split, (target_width, target_height), interp='bilinear')
                # img_resized = F.resize(img_split, target_height, target_width, interpolation=cv2.INTER_LINEAR)
                img_resized = F.resize(img_split, target_height, target_width, interpolation=cv2.INTER_LINEAR)

                # image transform, to torch float tensor 3xHxW
                img_resized = self.img_transform(img_resized)
                img_resized = torch.unsqueeze(img_resized, 0)
                img_resized_list.append(img_resized)
            img_resized_list = [x.contiguous() for x in img_resized_list]

            img_resized_split_list.append(img_resized_list)
        # avoid output original size, out of memory
        segm = F.resize(segm, int(ori_height / 2), int(ori_width / 2), interpolation=cv2.INTER_NEAREST)
        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = img_resized_split_list
        output['seg_label'] = batch_segms.contiguous()  # [1, 1824, 2432]
        #         print(output['seg_label'])
        output['info'] = this_record['fpath_img']

        return output

    def __len__(self):
        # return self.num_sample
        return len(self.list_sample)


class TestDataset_cv(BaseDataset_cv):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(TestDataset_cv, self).__init__(odgt, opt, **kwargs)
        # self.imgSizes = test_opt.imgSizes
        # self.imgMaxSize = test_opt.imgMaxSize
        self.root_dataset = root_dataset
        self.list_sample = self.list_sample[self.start_idx: self.end_idx]

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # h,w,c

        # ori_width, ori_height = img.size
        ori_height, ori_width = img.shape[:-1]

        img_resized_list = []
        for this_short_size in self.imgTestSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgTestMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            # img_resized = imresize(img, (target_width, target_height), interp='bilinear')
            img_resized = F.resize(img, target_height, target_width, interpolation=cv2.INTER_LINEAR)

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        # return self.num_sample
        return len(self.list_sample)


class TestDataset_Split_cv(BaseDataset_cv):
    def __init__(self, root_dataset, odgt, opt, image_split=2, **kwargs):
        super(TestDataset_Split_cv, self).__init__(odgt, opt, **kwargs)
        # self.imgSizes = test_opt.imgSizes
        # self.imgMaxSize = test_opt.imgMaxSize
        self.root_dataset = root_dataset
        self.list_sample = self.list_sample[self.start_idx: self.end_idx]
        self.image_split = image_split

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # h,w,c

        # ori_width, ori_height = img.size
        ori_height, ori_width = img.shape[:-1]

        img_split_coord_list = [(i, j) for i in range(self.image_split) for j in range(self.image_split)]

        img_ori_split_list = []
        img_resized_split_list = []
        for split_id in range(self.image_split ** 2):
            img_split_coord = img_split_coord_list[split_id]
            up_h = img_split_coord[0] * int(ori_height / self.image_split)
            left_w = img_split_coord[1] * int(ori_width / self.image_split)
            # img_split = TF.crop(img, up_h, left_w, int(ori_height / self.image_split),
            #                     int(ori_width / self.image_split))
            img_split = img[up_h:up_h + int(ori_height / self.image_split),
                        left_w:left_w + int(ori_width / self.image_split)]

            img_ori_split_list.append(np.array(img_split))

            img_resized_list = []
            for this_short_size in self.imgTestSizes:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            self.imgTestMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)

                # to avoid rounding in network
                target_width = self.round2nearest_multiple(target_width, self.padding_constant)
                target_height = self.round2nearest_multiple(target_height, self.padding_constant)

                # resize images
                # img_resized = imresize(img_split, (target_width, target_height), interp='bilinear')
                img_resized = F.resize(img_split, target_height, target_width, interpolation=cv2.INTER_LINEAR)

                # image transform, to torch float tensor 3xHxW
                img_resized = self.img_transform(img_resized)
                img_resized = torch.unsqueeze(img_resized, 0)
                img_resized_list.append(img_resized)
            img_resized_list = [x.contiguous() for x in img_resized_list]
            img_resized_split_list.append(img_resized_list)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = img_resized_split_list
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        # return self.num_sample
        return len(self.list_sample)

