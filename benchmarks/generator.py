# coding: utf-8

import numpy as np
from PIL import Image
from keras.utils import Sequence
from keras import backend as K

def _load_data(item_path):
    return Image.open(item_path)

class MyGenerator(Sequence):
    """Custom generator"""

    def __init__(self, data_paths, data_classes, 
                 batch_size=64, width=64, height=64, channel_num=3, num_of_class=10):
        """construction   

        :param data_paths: List of image file  
        :param data_classes: List of class  
        :param batch_size: Batch size  
        :param width: Image width  
        :param height: Image height  
        :param channel_num: Num of image channels  
        :param num_of_class: Num of classes  
        """

        #self.data_paths = data_paths
        #self.data_classes = data_classes
        if len(data_paths) != len(data_classes):
            raise ValueError('differ length between data_paths ({}) and data_classes ({}).'.format(len(data_paths), len(data_classes)))
        self.data_path_and_class = list(zip(data_paths, data_classes))
        self.length = len(data_paths)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.channel_num = channel_num
        self.num_of_class = num_of_class
        self.steps_per_epoch = int((self.length - 1) / batch_size) + 1


    def __getitem__(self, idx):
        """Get batch data   

        :param idx: Index of batch  

        :return imgs: numpy array of images 
        :return labels: numpy array of label  
        """

        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        items = self.data_path_and_class[start_pos : end_pos]
        imgs = np.empty((len(items), self.height, self.width, self.channel_num), dtype=np.float32)
        labels = np.empty((len(items), self.num_of_class), dtype=np.float32)

        for i, (item_path, item_class) in enumerate(items):
            img = _load_data(item_path)
            img = np.array(img.convert('L'))
            
            if K.image_data_format() == 'channels_first':
                img = img.reshape(self.channel_num, self.height, self.width)
            else:
                img = img.reshape(self.height, self.width, self.channel_num)

            img = img.astype('float32')
            img /= 255

            imgs[i, :] = img
            labels[i] = item_class 

        return imgs, labels


    def __len__(self):
        """Batch length"""

        return self.steps_per_epoch


    def on_epoch_end(self):
        """Task when end of epoch"""
        np.random.shuffle(self.data_path_and_class)

