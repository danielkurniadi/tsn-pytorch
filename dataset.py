import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint


def abs_listdir(directory, prefix=None):
    def abspath_join(directory, path):
        return os.path.abspath(os.path.join(directory, path))

    if prefix:
        filepaths = [
            abspath_join(directory, path)
            for path in os.listdir(directory)
            if path.startswith(prefix)
        ]

    else:
        filepaths = [
            abspath_join(directory, path)
            for path in os.listdir(directory)
        ]
    
    return filepaths


class VideoRecord(object):
    def __init__(self, row, custom = False, prefix=None):
        self._data = row
        self.custom = custom

        filepaths = abs_listdir(self._data[0], prefix=prefix)
        self.__num_frames = len(filepaths)

        if self.__num_frames == 0:
            print("No Frames %s, %s" %(prefix, self.path))

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        if self.custom:
            return int(self.__num_frames) - 9
        else:
            return int(self.__num_frames) - 1

    @property
    def label(self):
        return int(self._data[-1])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=1, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, custom_prefix = None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.custom_prefix = custom_prefix

        if custom_prefix:
            self.image_tmpl = self.custom_prefix + '{:05d}.jpg'
        else:
            self.image_tmpl = image_tmpl

        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            if idx != 1: # WHAT IS THIS FOR???
                idx = idx - 1
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        prefix = self.modality.lower()
        if prefix == 'flow':
            prefix = prefix + '_x'

        if self.image_tmpl.startswith('rp'):
            self.video_list = [VideoRecord(x.strip().split(' '), True, prefix=prefix) for x in open(self.list_file)]
        else:
            self.video_list = [VideoRecord(x.strip().split(' '), False, prefix=prefix) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
