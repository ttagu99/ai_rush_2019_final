from data_local_loader_keras import AiRushDataGenerator, build_cnn_model, make_history_distcnt, make_features_and_distcnt    
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
from sklearn.utils import class_weight
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
batch_size = 5000
def bind_nsml(feature_ext_model, model, task):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        feature_ext_model.save_weights(os.path.join(dir_name, 'feature_ext_model.h5'))
        model.save_weights(os.path.join(dir_name, 'model.h5'))
        print('model saved!')

    def load(dir_name):
        feature_ext_model.load_weights(os.path.join(dir_name, 'feature_ext_model.h5'))
        model.load_weights(os.path.join(dir_name, 'model.h5'))
        print('model loaded!')

    def infer(root, phase):
        return _infer(root, phase, model=model, task=task, feature_ext_model = feature_ext_model)

    nsml.bind(save=save, load=load, infer=infer)
    print('bind_nsml(model)')


def _infer(root, phase, model, task, feature_ext_model):
    print('_infer root - : ', root)
    csv_file = os.path.join(root, 'test', 'test_data', 'test_data')
    item = pd.read_csv(csv_file,
                            dtype={
                                'article_id': str,
                                'hh': int, 'gender': str,
                                'age_range': str,
                                'read_article_ids': str
                            }, sep='\t')

    print('item.shap', item.shape)
    print(item.head(10))

    article_list = item['article_id'].values.tolist()
    rm_dup_artilcle = list(set(article_list))
    history_sel_num = 1
    total_list_article=[]
    history_dupicate_top1=[]
    history_num = []
    log_num = 10000*10
    for cnt, idx in enumerate(item.index.to_list()):
        if(cnt%log_num==0):
            print('count process',cnt, '/',item.shape[0])
        cur_article = item['read_article_ids'].loc[idx]
        hist_top = "NoDup"
        if type(cur_article) == str:
            list_article = cur_article.split(',')
            total_list_article.extend(list_article)    
            for hist_article in list_article:
                if hist_article in rm_dup_artilcle:
                    hist_top = hist_article
                    break
        else:
            list_article = []
        history_dupicate_top1.append(hist_top)

        history_num.append(len(list_article))
    item['history_num'] = pd.Series(history_num, index=item.index)
    item['history_dupicate_top1'] = pd.Series(history_dupicate_top1, index=item.index)

    #only test set's article
    img_features, img_distcnts = make_features_and_distcnt(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image'),feature_ext_model
                                                                        ,article_list, 'features.pkl', 'distr_cnt.pkl')
    #only test history cnts
    history_distcnts = make_history_distcnt(total_list_article, 'history_distr_cnt.pkl')

    test_generator = AiRushDataGenerator( item, label=None,shuffle=False,batch_size=batch_size,mode='test'
                                             , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts)

    y_pred = []

    for i in item:
        y_pred.append(0.5)

    return y_pred
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

def build_model(input_feature_num):
    inp = Input(shape=(input_feature_num,))
    x = BatchNormalization(name = 'batchnormal_in')(inp)
    x = Dense(512, activation="relu")(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()     
    return model

def search_file(search_path):
    for subdir, dirs, files in os.walk(search_path):
        print(subdir,len(files))


class report_nsml(keras.callbacks.Callback):
    def __init__(self, prefix):
        'Initialization'
        self.prefix = prefix
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, epoch=epoch, loss=logs.get('loss'), val_loss=logs.get('val_loss'),acc=logs.get('acc'),val_acc=logs.get('val_acc'))
        nsml.save(self.prefix +'_' +str(epoch))

def main(args):   
    search_file(DATASET_PATH)
    feature_ext_model = build_cnn_model()
    model = build_model(2600)
    print('feature_ext_model.output.shape[1]',feature_ext_model.output.shape[1])
    if use_nsml:
        bind_nsml(feature_ext_model, model, args.task)
    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == 'train':
        csv_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')
        item = pd.read_csv(csv_file,
                                dtype={
                                    'article_id': str,
                                    'hh': int, 'gender': str,
                                    'age_range': str,
                                    'read_article_ids': str
                                }, sep='\t')
        print('item.shape', item.shape)
        print(item.head())

        label_data_path = os.path.join(DATASET_PATH, 'train',
                                        os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
        label = pd.read_csv(label_data_path,
                                    dtype={'label': int},
                                    sep='\t')
        print('train label csv')
        print(label.head())


        debug=1*10000
        if debug is not None:
            item= item[:debug]
            label = label[:debug]
        #class_weights = class_weight.compute_class_weight('balanced',  np.unique(label),   label)
        #print('class_weights',class_weights)

        article_list = item['article_id'].values.tolist()
        rm_dup_artilcle = list(set(article_list))
        history_sel_num = 1
        total_list_article=[]
        history_dupicate_top1=[]
        history_num = []
        log_num = 10000*10
        for cnt, idx in enumerate(item.index.to_list()):
            if(cnt%log_num==0):
                print('count process',cnt, '/',item.shape[0])
            cur_article = item['read_article_ids'].loc[idx]
            hist_top = "NoDup"
            if type(cur_article) == str:
                list_article = cur_article.split(',')
                total_list_article.extend(list_article)    
                for hist_article in list_article:
                    if hist_article in rm_dup_artilcle:
                        hist_top = hist_article
                        break
            else:
                list_article = []
            history_dupicate_top1.append(hist_top)

            history_num.append(len(list_article))
        item['history_num'] = pd.Series(history_num, index=item.index)
        item['history_dupicate_top1'] = pd.Series(history_dupicate_top1, index=item.index)
        print('preprocess item.shape', item.shape)
        print(item.head())
        print(item.columns)
        #only train set's article
        img_features, img_distcnts = make_features_and_distcnt(os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image'),feature_ext_model
                                                                            ,article_list, 'features.pkl', 'distr_cnt.pkl')
        #only train history cnts
        history_distcnts = make_history_distcnt(total_list_article, 'history_distr_cnt.pkl')
        train_df, valid_df, train_dfy, valid_dfy = train_test_split(item, label, test_size=0.05, random_state=777)#,stratify =label)
        print('train_df.shape, valid_df.shape, train_dfy.shape, valid_dfy.shape'
              ,train_df.shape, valid_df.shape, train_dfy.shape, valid_dfy.shape)
        # Generators
        #root=os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image')
        training_generator = AiRushDataGenerator( train_df, label=train_dfy,shuffle=False,batch_size=batch_size,mode='train'
                                                 , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts)
        validation_generator = AiRushDataGenerator( valid_df, label=valid_dfy,shuffle=False,batch_size=batch_size,mode='valid'
                                                  ,image_feature_dict=img_features,distcnts = img_distcnts,history_distcnts=history_distcnts)

        model.summary()

        """ Callback """
        monitor = 'val_loss'
        best_model_path = 'dgu_model.h5'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=5,factor=0.2,verbose=1)
        early_stop = EarlyStopping(monitor=monitor, patience=9)
        checkpoint = ModelCheckpoint(best_model_path,monitor=monitor,verbose=1,save_best_only=True)
        report = report_nsml(prefix = 'secls')
        callbacks = [reduce_lr,early_stop,checkpoint,report]


        # Train model on dataset
        model.fit_generator(generator=training_generator,   epochs=100, #class_weight=class_weights,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=4, callbacks=callbacks)
    #eda_set = next(training_generator)
    #print(len(eda_set), eda_set[0].shape, eda_set[1].shape)




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

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--task', type=str, default='ctrpred')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_epoch_every', type=int, default=2)
    parser.add_argument('--save_step_every', type=int, default=1000)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument("--arch", type=str, default="MLP")

    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)

    parser.add_argument('--dry_run', type=bool, default=False)

    config = parser.parse_args()
    main(config)