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
import tensorflow as tf
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
from efficientnet import EfficientNetB0
from keras.models import Model,load_model
from keras.optimizers import Adam, SGD
from keras.layers import Add,Multiply
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.python.keras import layers
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
batch_size = 2000
CNN_BACKBONE =MobileNetV2
debug=10000*40#None#10000#None#100000#None

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
    category_text_file = os.path.join(root, 'test', 'test_data', 'test_data_article.tsv') 
    category_text = pd.read_csv(category_text_file,
                            dtype={
                                'article_id': str,
                                'category_id': int,
                                'title': str
                            }, sep='\t')
    print('category_text.shape', category_text.shape)
    print(category_text.head())

    category_text = category_text[['article_id','category_id']]

    item,article_list,total_list_article = count_process(item, category_text)

    #only test set's article
    img_features, img_distcnts = make_features_and_distcnt(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image'),feature_ext_model
                                                                        ,article_list, 'features.pkl', 'distr_cnt.pkl')
    #only test history cnts
    history_distcnts = make_history_distcnt(total_list_article, 'history_distr_cnt.pkl')

    test_generator = AiRushDataGenerator( item, label=None,shuffle=False,batch_size=1,mode='test'
                                             , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts)

    y_pred =  model.predict_generator(test_generator)
    print('y_pred.shape', y_pred.shape)
    y_pred = y_pred.squeeze().tolist()
    print('y_pred list len',len(y_pred))
    #print(y_pred)
    return y_pred

def identity_block_1d(input_tensor, unit_num, drop_p=0.5):
    x = BatchNormalization()(input_tensor)
    x = Dense(unit_num, activation="relu")(x)
    x = Dropout(drop_p)(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def build_model(input_feature_num):
    inp = Input(shape=(input_feature_num,))
    x = BatchNormalization(name = 'batchnormal_in')(inp)
    x = Dense(512, activation="relu")(x)
    x = identity_block_1d(x,unit_num=512,drop_p=0.5)
    x = identity_block_1d(x,unit_num=512,drop_p=0.5)
    x = identity_block_1d(x,unit_num=512,drop_p=0.5)
    x = identity_block_1d(x,unit_num=512,drop_p=0.5)
    x = identity_block_1d(x,unit_num=512,drop_p=0.5)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model

def search_file(search_path):
    for subdir, dirs, files in os.walk(search_path):
        print(subdir,len(files))
        if len(files) <10:
            for file in files:
                print(file)

def check_history_func(cat, hist):
    if cat ==0 or hist ==0:
        return 0
    elif cat == hist:
        return 1
    else:
        return -1

def count_process(item, category_text):
    article_list = item['article_id'].values.tolist()
    rm_dup_artilcle = list(set(article_list))
    history_sel_num = 1
    total_list_article=[]
    history_dupicate_top1=[]
    history_num = []
    log_num = 10000*10
    
    history_left1=[]
    history_left2=[]
    history_left3=[]
    for cnt, idx in enumerate(item.index.to_list()):
        if(cnt%log_num==0):
            print('count process',cnt, '/',item.shape[0])
        cur_article = item['read_article_ids'].loc[idx]
        hist_top = "NoDup"
        if type(cur_article) == str:
            list_article = cur_article.split(',')
            if len(list_article)>=3:
                history_left3.append(list_article[2])
                history_left2.append(list_article[1])
                history_left1.append(list_article[0])
            elif len(list_article)==2:
                history_left3.append("")
                history_left2.append(list_article[1])
                history_left1.append(list_article[0])
            else:
                history_left3.append("")
                history_left2.append("")
                history_left1.append(list_article[0])

            total_list_article.extend(list_article)    
            for hist_article in list_article:  # so long~
                if hist_article in rm_dup_artilcle:
                    hist_top = hist_article
                    break
        else:
            history_left3.append("")
            history_left2.append("")
            history_left1.append("")
            list_article = []
        history_dupicate_top1.append(hist_top)

        history_num.append(len(list_article))
    item['history_num'] = pd.Series(history_num, index=item.index)
    #이미지 feature가 있는 history 1개
    item['history_dupicate_top1'] = pd.Series(history_dupicate_top1, index=item.index)
    ## 이미지 feature가 없는 history의 category top 2~3개 추가    
    if len(history_left1)!= len(history_left2) or  len(history_left1)!= len(history_left3):
        print('wrong history len',len(history_left1),len(history_left2),len(history_left3))

    item['history_left1'] = pd.Series(history_left1, index=item.index)
    item['history_left2'] = pd.Series(history_left2, index=item.index)
    item['history_left3'] = pd.Series(history_left3, index=item.index)
    
    item = pd.merge(item, category_text, how='left', on='article_id')
    print('merge item, category_text')
    print(item.head())
    category_text = category_text.rename(columns={"article_id": "context_article_id", "category_id": "history_category_id"})
    print('after rename category_text')
    print(category_text.head())
    item = pd.merge(item, category_text, how='left', left_on='history_dupicate_top1' , right_on= 'context_article_id')
    item = item.drop(columns=['context_article_id'])

    category_text = category_text.rename(columns={"history_category_id": "history_left1_category"})
    item = pd.merge(item, category_text, how='left', left_on='history_left1' , right_on= 'context_article_id')
    item = item.drop(columns=['context_article_id'])
    category_text = category_text.rename(columns={"history_left1_category": "history_left2_category"})
    item = pd.merge(item, category_text, how='left', left_on='history_left2' , right_on= 'context_article_id')
    item = item.drop(columns=['context_article_id'])
    category_text = category_text.rename(columns={"history_left2_category": "history_left3_category"})
    item = pd.merge(item, category_text, how='left', left_on='history_left3' , right_on= 'context_article_id')
    item = item.drop(columns=['context_article_id'])

    item['category_id'] =item['category_id'].fillna(value=0).astype(np.uint8)
    item['history_category_id'] =item['history_category_id'].fillna(value=0).astype(np.uint8)
    item['history_left1_category'] =item['history_left1_category'].fillna(value=0).astype(np.uint8)
    item['history_left2_category'] =item['history_left2_category'].fillna(value=0).astype(np.uint8)
    item['history_left3_category'] =item['history_left3_category'].fillna(value=0).astype(np.uint8)
    print('add history left category----------------------------------------------------------------------')
    print(item.head())

    check_category = []
    check_left1=[]
    check_left2=[]
    check_left3=[]
    cat_list = item['category_id'].tolist()
    hist_cat_list = item['history_category_id'] .tolist()
    hist_left1_list =item['history_left1_category'].tolist()
    hist_left2_list =item['history_left2_category'].tolist() 
    hist_left3_list =item['history_left3_category'].tolist() 
    for i in range(len(cat_list)):
        check_category.append(check_history_func(cat_list[i], hist_cat_list[i]))
        check_left1.append(check_history_func(cat_list[i], hist_left1_list[i]))
        check_left2.append(check_history_func(cat_list[i], hist_left3_list[i]))
        check_left3.append(check_history_func(cat_list[i], hist_left3_list[i]))

    item['check_category'] = check_category
    item['check_left1'] = check_left1
    item['check_left2'] = check_left2
    item['check_left3'] = check_left3
    return item,article_list,total_list_article

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def KerasFocalLoss(target, input):
    gamma = 2.
    input = tf.cast(input, tf.float32)
    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    return K.mean(K.sum(loss, axis=1))

class report_nsml(keras.callbacks.Callback):
    def __init__(self, prefix):
        'Initialization'
        self.prefix = prefix
    def on_epoch_end(self, epoch, logs={}):
        nsml.report(summary=True, epoch=epoch, loss=logs.get('loss'), val_loss=logs.get('val_loss')
                    ,acc=logs.get('acc'),val_acc=logs.get('val_acc')
                    ,f1_score=logs.get('f1_score'),val_f1_score=logs.get('val_f1_score'))
        nsml.save(self.prefix +'_' +str(epoch))

def main(args):   
    search_file(DATASET_PATH)
    if args.mode == 'train':
        feature_ext_model = build_cnn_model(backbone=CNN_BACKBONE)
    else:
        feature_ext_model = build_cnn_model(backbone=CNN_BACKBONE,use_imagenet=None)

    in_feature_num = int(97 +84 + 9+ feature_ext_model.output.shape[1]*2)
    print( 'in_feature_num',in_feature_num)
    model = build_model(in_feature_num)
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
        category_text_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data_article.tsv')

 
        category_text = pd.read_csv(category_text_file,
                                dtype={
                                    'article_id': str,
                                    'category_id': int,
                                    'title': str
                                }, sep='\t')
        print('category_text.shape', category_text.shape)
        print(category_text.head())

        category_text = category_text[['article_id','category_id']]
        

        print('category_id].values.max()',category_text['category_id'].values.max())
        print('category_id].values.min()',category_text['category_id'].values.min())

        label_data_path = os.path.join(DATASET_PATH, 'train',
                                        os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')
        label = pd.read_csv(label_data_path,
                                    dtype={'label': int},
                                    sep='\t')
        print('train label csv')
        print(label.head())



        if debug is not None:
            item= item[:debug]
            label = label[:debug]
        #class_weights = class_weight.compute_class_weight('balanced',  np.unique(label),   label)
        #print('class_weights',class_weights)
        item,article_list,total_list_article = count_process(item,category_text)
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
        training_generator = AiRushDataGenerator( train_df, label=train_dfy,shuffle=True,batch_size=batch_size,mode='train'
                                                 , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts
                                                 ,featurenum=in_feature_num)
        validation_generator = AiRushDataGenerator( valid_df, label=valid_dfy,shuffle=False,batch_size=batch_size//20,mode='valid'
                                                  ,image_feature_dict=img_features,distcnts = img_distcnts,history_distcnts=history_distcnts
                                                  ,featurenum=in_feature_num)


        metrics=['accuracy',f1_score]

        opt = Adam(lr=0.001)
        model.compile(loss=f1_loss, optimizer=opt, metrics=metrics)
        model.summary()

        """ Callback """
        monitor = 'val_f1_score'
        best_model_path = 'dgu_model.h5'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=5,factor=0.2,verbose=1,mode='max')
        early_stop = EarlyStopping(monitor=monitor, patience=9,mode='max')

        #checkpoint = ModelCheckpoint(best_model_path,monitor=monitor,verbose=1,save_best_only=True)
        report = report_nsml(prefix = 'dgu')
        callbacks = [reduce_lr,early_stop,report]

        # Train model on dataset
        model.fit_generator(generator=training_generator,steps_per_epoch=100,   epochs=10000, #class_weight=class_weights,
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