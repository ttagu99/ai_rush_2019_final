from data_local_loader_keras import get_data_loader
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision.models as models
import os
import argparse
import numpy as np
import time
import datetime

from data_loader import feed_infer
from evaluation import evaluation_metrics
import nsml
import keras
from keras.models import Sequential
from keras.layers import Concatenate
from keras.layers import Dense, Dropout, Flatten, Activation,Average
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model,load_model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
#import imgaug as ia
#from imgaug import augmenters as iaa
#import lightgbm as lgb
from sklearn.externals import joblib
from lightgbm import LGBMClassifier


# expected to be a difficult problem
# Gives other meta data (gender age, etc.) but it's hard to predict click through rate
# How to use image and search history seems to be the key to problem solving. Very important data
# Image processing is key. hint: A unique image can be much smaller than the number of data.
# For example, storing image features separately and stacking them first,
# then reading them and learning artificial neural networks is good in terms of GPU efficiency.
# -> image feature has been extracted and loaded separately.
# The retrieval history is how to preprocess the sequential data and train it on which model.
# Greatly needed efficient coding of CNN RNNs.
# You can also try to change the training data set itself. Because it deals with very imbalanced problems.
# Refactor to summarize from existing experiment code.

DATASET_PATH = os.path.join(nsml.DATASET_PATH)
print('start using nsml...!')
print('DATASET_PATH: ', DATASET_PATH)
use_nsml = True

def bind_nsml(cnn_model,gbm_model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        cnn_model.save_weights(os.path.join(dir_name, 'cnn_model.h5'))
        print('cnn_model saved!', os.path.join(dir_name, 'cnn_model.h5'))
        joblib.dump(gbm_model,  os.path.join(dir_name, 'gbm_model.pkl'))
        print('gbm_model saved!', os.path.join(dir_name, 'gbm_model.pkl'))

    def load(dir_name):
        cnn_model.load_weights(os.path.join(dir_name, 'cnn_model.h5'))
        print('cnn_model loaded!', os.path.join(dir_name, 'cnn_model.h5'))
        gbm_model = joblib.load( os.path.join(dir_name, 'gbm_model.pkl'))
        print('gbm_model loaded!',  os.path.join(dir_name, 'gbm_model.pkl'))
        print('loaded model checkpoints...!')

    def infer(root):
        pass

    nsml.bind(save=save, load=load, infer=infer)
    print('bind_nsml(cnn_model,gbm_model)')


def _infer(root, phase, model, task):
    pass
    # root : csv file path
    #print('_infer root - : ', root)
    #with torch.no_grad():
    #    model.eval()
    #    test_loader, dataset_sizes = get_data_loader(root, phase)
    #    y_pred = []
    #    print('start infer')
    #    for i, data in enumerate(test_loader):
    #        images, extracted_image_features, labels, flat_features = data

    #        # images = images.cuda()
    #        extracted_image_features = extracted_image_features.cuda()
    #        flat_features = flat_features.cuda()
    #        # labels = labels.cuda()

    #        logits = model(extracted_image_features, flat_features)
    #        y_pred += logits.cpu().squeeze().numpy().tolist()

    #    print('end infer')
    #return y_pred


def build_cnn_model(backbone= MobileNetV2, input_shape =  (224,224,3), use_imagenet = 'imagenet', base_freeze=True):
    base_model = backbone(input_shape=input_shape, weights=use_imagenet, include_top= False)#, classes=NCATS)
    x = base_model.output
    gap_x = GlobalAveragePooling2D()(x)
    #predict = Dense(num_classes, activation='softmax', name='last_softmax')(x)
    model = Model(inputs=base_model.input, outputs=gap_x)
    if base_freeze==True:
        for layer in base_model.layers:
            layer.trainable = False
    #model.compile(loss='categorical_crossentropy',   optimizer=opt,  metrics=['accuracy'])
    print('build_cnn_model')
    return model

def main(args):   
    cnn_model = build_cnn_model(backbone=MobileNetV2, use_imagenet = None)
    gbm_model = LGBMClassifier(objective='binary', random_state=777)

    if use_nsml:
        bind_nsml(cnn_model, gbm_model)
    if args.pause:
        nsml.paused(scope=locals())

    if (args.mode == 'train'):
        train_loader, dataset_sizes = get_data_loader(
            root=os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data'),
            phase='train',
            batch_size=args.batch_size)

        start_time = datetime.datetime.now()
        iter_per_epoch = len(train_loader)
        #best_loss = 1000
        #if args.dry_run:
        #    print('start dry-running...!')
        #    args.num_epochs = 1
        #else:
        #    print('start training...!')

        for epoch in range(args.num_epochs):
            for i, data in enumerate(train_loader):
                images, extracted_image_features, labels, flat_features = data

                images = images.cuda()
                extracted_image_features = extracted_image_features.cuda()
                flat_features = flat_features.cuda()
                labels = labels.cuda()

                # forward
                if args.arch == 'MLP':
                    logits = model(extracted_image_features, flat_features)
                elif args.arch == 'Resnet':
                    logits = model(images, flat_features)
                criterion = nn.MSELoss()
                loss = torch.sqrt(criterion(logits.squeeze(), labels.float()))

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < best_loss:
                    nsml.save('best_loss')  # this will save your best model on nsml.



                if i % args.print_every == 0:
                    elapsed = datetime.datetime.now() - start_time
                    print('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f'
                          % (elapsed, epoch + 1, args.num_epochs, i + 1, iter_per_epoch, loss.item()))
                #if i % args.save_step_every == 0:
                #    # print('debug ] save testing purpose')
                #    nsml.save('step_' + str(i))  # this will save your current model on nsml.

            if epoch % args.save_epoch_every == 0:
                nsml.report(
                    summary=True,
                    step=epoch,
                    scope=locals(),
                    **{
                    "Loss": loss.item(),
                    })

                nsml.save('epoch_' + str(epoch))  # this will save your current model on nsml.
    nsml.save('final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)  # not work. check built_in_args in data_local_loader.py

    parser.add_argument('--train_path', type=str, default='train/train_data/train_data')
    parser.add_argument('--test_path', type=str, default='test/test_data/test_data')
    parser.add_argument('--test_tf', type=str, default='[transforms.Resize((456, 232))]')
    parser.add_argument('--train_tf', type=str, default='[transforms.Resize((456, 232))]')

    parser.add_argument('--use_sex', type=bool, default=True)
    parser.add_argument('--use_age', type=bool, default=True)
    parser.add_argument('--use_exposed_time', type=bool, default=True)
    parser.add_argument('--use_read_history', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=10)#1)
    parser.add_argument('--batch_size', type=int, default=350)#2048)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_epoch_every', type=int, default=1)
    parser.add_argument('--save_step_every', type=int, default=1000)#)1000)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="Resnet")#"MLP")#"Resnet")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)#True)#False)

    config = parser.parse_args()
    main(config)
