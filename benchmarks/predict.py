#!/usr/bin/env python3
# coding: utf-8

import argparse
from pathlib import Path
import pickle
import sys
from statistics import mean
from PIL import Image

import keras
from keras import backend as K
import numpy as np
import tqdm

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="2", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

def run(args):
    lb_model = pickle.load(args.lb_model)

    model = keras.models.load_model(args.model_file) 

    args.output_file.write('image_id,labels\n')

    for input_img_path in Path(args.input_img_dir).glob('*.jpg'):
        input_box_path = Path(args.input_box_dir) / input_img_path.with_suffix('.txt').name
        if not input_box_path.exists():
            args.output_file.write(input_img_path.stem + ',\n')
            continue

        results = predict(input_img_path, input_box_path, model, lb_model)
        args.output_file.write(Path(input_img_path).stem + ',' + results + '\n')
        args.output_file.flush()


def predict(input_img_path, input_box_path, model, lb_model):

    crops = []
    coords = []
    im = Image.open(input_img_path)
    im = im.convert('L')
    with input_box_path.open(mode='r') as f_input_box:
        for line in f_input_box:
            c = [int(c) for c in line.strip().split(',')]
            im_crop = im.crop((c[0], c[1], c[4], c[5]))

            im_crop = np.array(im_crop.resize((64, 64)), dtype='float32') / 255

            if K.image_data_format() == 'channels_first':
                im_crop = im_crop.reshape(1, 64, 64)
            else:
                im_crop = im_crop.reshape(64, 64, 1)

            crops.append(im_crop) 
            coords.append((int(mean([c[0], c[4]])), int(mean([c[1], c[5]]))))
       
    if not crops:
        return ''

    crops = np.array(crops)

    pred_y = model.predict(crops)
    pred_labels = lb_model.inverse_transform(pred_y)

    assert len(coords) == len(pred_labels)

    results = ' '.join([' '.join([label, str(coord[0]), str(coord[1])])
                            for label, coord in zip(pred_labels, coords)])
    return results        


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_img_dir', type=str)
    parser.add_argument('input_box_dir', type=str)
    parser.add_argument('model_file', type=str)
    parser.add_argument('-l', '--label-binarizer-model', dest='lb_model',
                        type=argparse.FileType('rb'),
                        default='./label_binarizer.model')
    parser.add_argument('-o', '--output-file', dest='output_file',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    return parser.parse_args()

def main():
    args = arg_parse()
    run(args)

if __name__ == '__main__':
    main()
