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
batch_size = 2000
CNN_BACKBONE =MobileNetV2
# CNN_BACKBONE = Xception
start_list = 173000
debug=346000#None
use_ensemble_num = 6
#ens_weight=[0.4,0.4,0.4, 1]
#ens_weight /= np.sum(ens_weight)
#print(ens_weight)
def median(data):
    new_list = sorted(data)
    if len(new_list)%2 > 0:
        return new_list[int(len(new_list)/2)]
    elif len(new_list)%2 == 0:
        return (new_list[int((len(new_list)/2))] + new_list[int((len(new_list)/2)-1)]) /2.0

from keras.callbacks import *


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


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

    item,article_list,total_list_article = count_process(item)

    #only test set's article
    img_features, img_distcnts = make_features_and_distcnt(os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image'),feature_ext_model
                                                                        ,article_list, 'features.pkl', 'distr_cnt.pkl')
    #only test history cnts
    history_distcnts = make_history_distcnt(total_list_article, 'history_distr_cnt.pkl')

    test_generator = AiRushDataGenerator( item, label=None,shuffle=False,batch_size=1,mode='test'
                                             , image_feature_dict=img_features,distcnts = img_distcnts, history_distcnts=history_distcnts)


    y_pred = []
    for i in range(item.shape[0]):
        cur_x, cur_y = test_generator.__getitem__(i)
        inputs =[]
        for mm in range(use_ensemble_num):
            inputs.append(cur_x)
                    
        probs = model.predict(inputs)
        #for idx,prob in enumerate(probs):
        #    probs[idx] = prob*ens_weight[idx]
        thr=0.5
        if np.mean(probs) > thr:
            y_pred.append(1.0)
        else:
            y_pred.append(0.0)
        #print('probs',probs,'median', median(probs))
    #print('y_pred.shape', y_pred.shape)
    #y_pred = y_pred.squeeze().tolist()
    print(y_pred)
    print('y_pred list len',len(y_pred))
    #print(y_pred)
    return y_pred


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
    return model

def search_file(search_path):
    for subdir, dirs, files in os.walk(search_path):
        print(subdir,len(files))


def count_process(item):
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
            for hist_article in list_article:  # so long~
                if hist_article in rm_dup_artilcle:
                    hist_top = hist_article
                    break
        else:
            list_article = []
        history_dupicate_top1.append(hist_top)

        history_num.append(len(list_article))
    item['history_num'] = pd.Series(history_num, index=item.index)
    item['history_dupicate_top1'] = pd.Series(history_dupicate_top1, index=item.index)
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

    #개별 모델 정보
#nsml.load(checkpoint='193_base_611', session='team_27/airush2/229')
#nsml.load(checkpoint='193_base_202', session='team_27/airush2/645')
#nsml.load(checkpoint='part_03_114', session='team_27/airush2/671')
#nsml.load(checkpoint='part_03_146', session='team_27/airush2/673')
#nsml.load(checkpoint='part_03_94', session='team_27/airush2/684')
#nsml.load(checkpoint='part_03_57', session='team_27/airush2/688')


    model1 = build_model(2600)
    model2 = build_model(2600)  
    model3 = build_model(2600)  
    model4 = build_model(2600)  
    model5 = build_model(2600)
    model6 = build_model(2600)
    print('feature_ext_model.output.shape[1]',feature_ext_model.output.shape[1])
    if use_nsml:
        bind_nsml(feature_ext_model, model1, args.task)
        nsml.load(checkpoint='193_base_611', session='team_27/airush2/229') 
        bind_nsml(feature_ext_model, model2, args.task)
        nsml.load(checkpoint='193_base_202', session='team_27/airush2/645')
        bind_nsml(feature_ext_model, model3, args.task)
        nsml.load(checkpoint='part_03_114', session='team_27/airush2/671')
        bind_nsml(feature_ext_model, model4, args.task)
        nsml.load(checkpoint='part_03_146', session='team_27/airush2/673')
        bind_nsml(feature_ext_model, model5, args.task)
        nsml.load(checkpoint='part_03_94', session='team_27/airush2/684')
        bind_nsml(feature_ext_model, model6, args.task)
        nsml.load(checkpoint='part_03_57', session='team_27/airush2/688')
        #nsml.load(checkpoint='part_03_21', session='team_27/airush2/671')
        #bind_nsml(feature_ext_model, model3, args.task)
        #nsml.load(checkpoint='part_03_24', session='team_27/airush2/673')


        merge_model = Model(inputs=[model1.input, model2.input,model3.input, model4.input , model5.input, model6.input]
                            , outputs=[model1.output, model2.output,model3.output, model4.output,  model5.output, model6.output ])
        bind_nsml(feature_ext_model, merge_model, args.task)
        nsml.save('dgu_final')

        # megrging


    if args.pause:
        nsml.paused(scope=locals())
 



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
    parser.add_argument('--lr', type=float, default=0.0001)
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
