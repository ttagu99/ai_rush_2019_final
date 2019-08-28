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
from sklearn.externals import joblib
import lightgbm as lgb
import gc
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score as f1_score_sk
from keras_main import count_process

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True
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
#batch_size = 250000
CNN_BACKBONE =MobileNetV2
debug=200*10000#None#10000#None#100000#None
use_image_feature= True
#if use_image_feature == False:
#    debug=None
balancing = True

def bind_nsml(cnn_model,gbm_model, task):
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

    def infer(root, phase):
        return _infer(root, phase, gbm_model=gbm_model, task=task, feature_ext_model = cnn_model)

    nsml.bind(save=save, load=load, infer=infer)
    print('bind_nsml(cnn_model,gbm_model)')


def _infer(root, phase, gbm_model, task, feature_ext_model):
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
    if use_image_feature==True:
       in_feature_num = int(97 +84 + 9+ feature_ext_model.output.shape[1]*2)
    else:
       in_feature_num = int(97 +84 + 9)

    #only test set's article
    img_features, img_distcnts = make_features_and_distcnt(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image'),feature_ext_model
                                                                        ,article_list, 'features.pkl', 'distr_cnt.pkl')
    #only test history cnts
    history_distcnts = make_history_distcnt(total_list_article, 'history_distr_cnt.pkl')

    test_generator = AiRushDataGenerator( item, label=None,shuffle=False,batch_size=item.shape[0],mode='test'
                                             , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts
                                             ,featurenum=in_feature_num,use_image_feature=use_image_feature)

    X, y = test_generator.__getitem__(0) 
    print('X.shape', X.shape)
    y_pred = gbm_model.predict(X)
    #y_pred =  model.predict_generator(test_generator)
    print('y_pred.shape', y_pred.shape)
    y_pred = y_pred.squeeze().tolist()
    print('y_pred list len',len(y_pred))
    return y_pred

def identity_block_1d(input_tensor, unit_num, drop_p=0.5):
    x = BatchNormalization()(input_tensor)
    x = Dense(unit_num, activation="relu")(x)
    x = Dropout(drop_p)(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def attention_block(input_tensor,unit_num):
    attention_probs = Dense(unit_num, activation='softmax')(input_tensor)
    attention_mul = Multiply()([input_tensor, attention_probs])
    return attention_mul




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


def evaluation(true,pred):
   true[true == 0] = -1
   true[true == 1] = 1
   p = np.average(pred) #probability.
   normalized_entropy = 0
   exception_value = -1
   left_vec = np.average(np.dot((1 + true)/2, [np.log(p) if p > 0 else exception_value for p in pred]))
   right_vec = np.average(np.dot((1 - true)/2, [np.log(1 - p) if (1-p) > 0 else exception_value for p in pred]))
   base = -1*(p * (math.log(p) if p > 0 else exception_value) + (1-p)*(math.log(1-p) if 1-p > 0 else exception_value))
   normalized_entropy = (left_vec + right_vec)*(-1)
   normalized_entropy = normalized_entropy / (base if base != 0 else 1)
   return normalized_entropy #not normalized well. the range can be larger than 1

class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []
    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
        self._data.append({
            'val_pctr': evaluation(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data


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


    #gbm_model = lgb.LGBMClassifier()#lgb.Booster()
    print("Initial Training the model...")
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_error',
        'learning_rate': 0.0015,
        'num_leaves': 255,  
        'max_depth': -1,  
        'min_child_samples': 1000,#1000,  
        'max_bin': 100,  
        'subsample': 0.8,  
        'subsample_freq': 1,  
        'colsample_bytree': 0.8,  
        'min_child_weight': 0,  
        'subsample_for_bin': 10000,#20000,  
        'min_split_gain': 0,  
        'reg_alpha': 0,  
        'reg_lambda': 0,  
        # 'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':1
        }
    evals_results = {}
    TrainX = ValidX = np.zeros((100,2))
    TrainY = ValidY = np.zeros((100))
    dtrain = lgb.Dataset(TrainX, label=TrainY)
    dvalid = lgb.Dataset(ValidX,  label=ValidY)
    gbm_model = lgb.train(params, 
                            dtrain, 
                            valid_sets=[dtrain, dvalid], 
                            valid_names=['train','valid'], 
                            evals_result=evals_results, 
                            num_boost_round=1,
                            early_stopping_rounds=1,
                            verbose_eval=True, 
                            feval=None)


    if args.mode == 'train':
        feature_ext_model = build_cnn_model(backbone=CNN_BACKBONE)
    else:
        feature_ext_model = build_cnn_model(backbone=CNN_BACKBONE,use_imagenet=None)

    if use_image_feature==True:
        in_feature_num = int(97 +84 + 9+ feature_ext_model.output.shape[1]*2)
    else:
       in_feature_num = int(97 +84 + 9)
    print( 'in_feature_num',in_feature_num)
    #model = build_model(in_feature_num)
    print('feature_ext_model.output.shape[1]',feature_ext_model.output.shape[1])

    #def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1,
    #             learning_rate=0.1, n_estimators=100,
    #             subsample_for_bin=200000, objective=None, class_weight=None,
    #             min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
    #             subsample=1., subsample_freq=0, colsample_bytree=1.,
    #             reg_alpha=0., reg_lambda=0., random_state=None,
    #             n_jobs=-1, silent=True, importance_type='split', **kwargs):




    if use_nsml:
        bind_nsml(feature_ext_model, gbm_model, args.task)
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

        if balancing == True:
            one_label = label[label['label']==1]
            print(one_label.head())
            zero_label = label[label['label']==0].sample(one_label.shape[0])
            print(zero_label.head())
            label = pd.concat([one_label,zero_label])
            #print(label.index.to_list())
            item = item.loc[label.index.to_list()]
            print('item.shape',item.shape)
            print(item.head())
            print(label.head())

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
        training_generator = AiRushDataGenerator( train_df, label=train_dfy,shuffle=True,batch_size=train_df.shape[0],mode='train'
                                                 , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts
                                                 ,featurenum=in_feature_num,use_image_feature=use_image_feature)
        validation_generator = AiRushDataGenerator( valid_df, label=valid_dfy,shuffle=False,batch_size=valid_df.shape[0],mode='valid'
                                                  ,image_feature_dict=img_features,distcnts = img_distcnts,history_distcnts=history_distcnts
                                                  ,featurenum=in_feature_num,use_image_feature=use_image_feature)

        #pctr = Metrics()#next(training_generator.flow())
        print('make train data')
        TrainX, TrainY = training_generator.__getitem__(0)
        print('make valid data')
        ValidX, ValidY = validation_generator.__getitem__(0)
        print('train valid shape',TrainX.shape, TrainY.shape, ValidX.shape, ValidY.shape)        


        print("Training the model...")
        dtrain = lgb.Dataset(TrainX, label=TrainY)
        del TrainX
        del TrainY
        gc.collect()
        dvalid = lgb.Dataset(ValidX,  label=ValidY)
        ValidX
        ValidY
        gc.collect()

        gbm_model = lgb.train(params, 
                         dtrain,
                         valid_sets=[dtrain, dvalid], 
                         valid_names=['train','valid'], 
                         evals_result=evals_results, 
                         num_boost_round=5000,
                         early_stopping_rounds=30,
                         #feval=lgb_f1_score,
                         verbose_eval=True)
        validPred = gbm_model.predict(ValidX)
        #print('f1_score_sk',f1_score_sk(ValidY,validPred))
        print(validPred)
        nsml.save('dgu_sample')

        



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
