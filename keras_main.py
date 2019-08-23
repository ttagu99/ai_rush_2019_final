from data_local_loader_keras import AiRushDataGenerator
import os
import argparse
import numpy as np
import time
import datetime
import pandas as pd

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
from nsml import DATASET_PATH, DATASET_NAME, NSML_NFS_OUTPUT, SESSION_NAME
#import imgaug as ia
#from imgaug import augmenters as iaa
#import lightgbm as lgb
#from sklearn.externals import joblib
#from lightgbm import LGBMClassifier
## sklearn tools for model training and assesment
#from sklearn.model_selection import GridSearchCV

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

def bind_nsml(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model.h5'))
        print('model saved!', os.path.join(dir_name, 'model.h5'))
        #joblib.dump(gbm_model,  os.path.join(dir_name, 'gbm_model.pkl'))
        #print('gbm_model saved!', os.path.join(dir_name, 'gbm_model.pkl'))
    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model.h5'))
        print('model loaded!', os.path.join(dir_name, 'model.h5'))
        #gbm_model = joblib.load( os.path.join(dir_name, 'gbm_model.pkl'))
        #print('gbm_model loaded!',  os.path.join(dir_name, 'gbm_model.pkl'))
        print('loaded model checkpoints...!')

    def infer(root):
        pass

    nsml.bind(save=save, load=load, infer=infer)
    print('bind_nsml(cnn_model,gbm_model)')


def _infer(root, phase, model, task):
    csv_file = os.path.join(csv_file, 'test', 'test_data', 'test_data')
    item = pd.read_csv(csv_file,
                            dtype={
                                'article_id': str,
                                'hh': int, 'gender': str,
                                'age_range': str,
                                'read_article_ids': str
                            }, sep='\t')

    print('item.shap', item.shape)
    print(item.head(10))

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

def build_model():
    inp = Input(shape=(4048,1))
    x = Dense(128, activation="relu")(inp)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()     
    return model

def main(args):   
    model = build_model()
    print(model.output.shape)
    if use_nsml:
        bind_nsml(model)
    if args.pause:
        nsml.paused(scope=locals())


    csv_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')
    item = pd.read_csv(csv_file,
                            dtype={
                                'article_id': str,
                                'hh': int, 'gender': str,
                                'age_range': str,
                                'read_article_ids': str
                            }, sep='\t')

    print('item.shap', item.shape)
    print(item.head(10))

    label_data_path = os.path.join(DATASET_PATH, 'train',
                                    os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
    label = pd.read_csv(label_data_path,
                                dtype={'label': int},
                                sep='\t')
    print('train label csv')
    print(label.head(10))



    train_df, valid_df, train_dfy, valid_dfy = train_test_split(item, label, test_size=0.05, random_state=777,stratify =label)
    print('train_df.shape, valid_df.shape, train_dfy.shape, valid_dfy.shape'
          ,train_df.shape, valid_df.shape, train_dfy.shape, valid_dfy.shape)
    # Generators
    root=os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image')
    training_generator = AiRushDataGenerator(root, train_df, label=train_dfy,shuffle=False,batch_size=200,mode='train')
    validation_generator = AiRushDataGenerator(root, valid_df, label=valid_dfy,shuffle=False,batch_size=200,mode='train')

    model.summary()

    eda_set = next(training_generator)
    print(len(eda_set), eda_set[0].shape, eda_set[1].shape)


    nsml.save('last')
        ## Create the grid
        #grid = GridSearchCV(gbm_model, gridParams,
        #                    verbose=1,
        #                    cv=4,
        #                    n_jobs=2)
        ## Run the grid
        #grid.fit(TotalX, TotalY)

        ## Print the best parameters found
        #print(grid.best_params_)
        #print(grid.best_score_)

    #    for epoch in range(args.num_epochs):
    #        for i, data in enumerate(train_loader):
    #            images, extracted_image_features, labels, flat_features = data

    #            images = images.cuda()
    #            extracted_image_features = extracted_image_features.cuda()
    #            flat_features = flat_features.cuda()
    #            labels = labels.cuda()

    #            # forward
    #            if args.arch == 'MLP':
    #                logits = model(extracted_image_features, flat_features)
    #            elif args.arch == 'Resnet':
    #                logits = model(images, flat_features)
    #            criterion = nn.MSELoss()
    #            loss = torch.sqrt(criterion(logits.squeeze(), labels.float()))

    #            # backward and optimize
    #            optimizer.zero_grad()
    #            loss.backward()
    #            optimizer.step()

    #            if loss < best_loss:
    #                nsml.save('best_loss')  # this will save your best model on nsml.



    #            if i % args.print_every == 0:
    #                elapsed = datetime.datetime.now() - start_time
    #                print('Elapsed [%s], Epoch [%i/%i], Step [%i/%i], Loss: %.4f'
    #                      % (elapsed, epoch + 1, args.num_epochs, i + 1, iter_per_epoch, loss.item()))
    #            #if i % args.save_step_every == 0:
    #            #    # print('debug ] save testing purpose')
    #            #    nsml.save('step_' + str(i))  # this will save your current model on nsml.

    #        if epoch % args.save_epoch_every == 0:
    #            nsml.report(
    #                summary=True,
    #                step=epoch,
    #                scope=locals(),
    #                **{
    #                "Loss": loss.item(),
    #                })

    #            nsml.save('epoch_' + str(epoch))  # this will save your current model on nsml.
    #nsml.save('final')


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
