import pickle
import os
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
import nsml
from utils import get_transforms
from utils import default_loader,pil_loader
from collections import Counter
import operator
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
import random
from keras.preprocessing import image
from keras.applications.mobilenetv2 import preprocess_input


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
    model.summary()
    return model

def merge_list(image_list):
    result = Counter(image_list)
    result = sorted(result.items(), key=operator.itemgetter(1),reverse=False)
    print('merege_list')
    return result

def extract_feature(model, image_path):
    try:
        img = pil_loader(image_path)
        img = img.resize((224, 224))
        img_data = image.img_to_array(img)
    except:
        img_data = np.zeros((224,224,3))
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    return feature

def make_history_distcnt(full_list, distcnt_path):
    image_dict = merge_list(full_list)
    distcnts =dict()
    total_len = len(image_dict)
    prs=0
    for article_id, cnt in image_dict:
        distcnts[article_id] = cnt
        prs+=1
    output = open(distcnt_path, 'wb')
    pickle.dump(distcnts, output)
    output.close()
    return distcnts



def make_features_and_distcnt(root_dir, model, full_list, save_path, distcnt_path):
    image_dict = merge_list(full_list)
    features=dict()
    distcnts =dict()
    total_len = len(image_dict)
    prs=0
    for article_id, cnt in image_dict:
        if (prs%100==0):
            print('extract features process:',prs,'/',total_len)
        img_name = os.path.join(root_dir, article_id + '.jpg')
        cur_feature = extract_feature(model,img_name)
        features[article_id] = cur_feature
        distcnts[article_id] = cnt
        prs+=1

    output = open(save_path, 'wb')
    pickle.dump(features, output)
    output.close()
    output = open(distcnt_path, 'wb')
    pickle.dump(distcnts, output)
    output.close()
    return features, distcnts






if not nsml.IS_ON_NSML:
    # if you want to run it on your local machine, then put your path here
    DATASET_PATH = '/temp'
    print(DATASET_PATH)
    DATASET_NAME = 'airush2_temp'
else:
    from nsml import DATASET_PATH, DATASET_NAME, NSML_NFS_OUTPUT, SESSION_NAME

    print('DATASET_PATH: ', DATASET_PATH)



class AiRushDataGenerator(keras.utils.Sequence):
    def __init__(self,  root_dir,  item, label=None,
                 transform=None,shuffle=False,batch_size=200,mode='train' #, features_model=None
                 , image_feature_dict=None, distcnts=None, history_distcnts = None
                 ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label = label
        self.item = item
        self.indexes = self.item.index.values.tolist()
        #self.n = 0
        #self.max = self.item.shape[0]//self.batch_size
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.hist_maxuse_num = 1
        #self.features_model = features_model
        self.on_epoch_end()
        self.image_feature_dict = image_feature_dict
        self.distcnts =distcnts
        self.history_distcnts = history_distcnts
        self.sex = {'unknown': 0, 'm': 1, 'f': 2}
        self.age = {'unknown': 0, '-14': 1, '15-19': 2, '20-24': 3, '25-29': 4,
                        '30-34': 5, '35-39': 6, '40-44': 7, '45-49': 8, '50-': 9}

        for c in self.item.columns:
            print(c)
            print(self.item[c].head(10))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.item) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(idxs)
        return X, y

    def __data_generation(self, idxs):
        X = np.empty((self.batch_size, 2600))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, idx in enumerate(idxs):
            # Store sample
            X[i,] ,  y[i] = self.get_one_data(idx)
        return X,y

    
    def get_one_data(self, idx):
        article_id, hh, gender, age_range, read_article_ids,history_num,history_dupicate_top1  = self.item.loc[idx
                       , ['article_id', 'hh', 'gender', 'age_range', 'read_article_ids','history_num','history_dupicate_top1']]

        #if self.mode== 'train' or self.mode=='valid':
        label = self.label.loc[idx,['label']]
        label = np.array(label, dtype=np.float32)
        #else:
        #    # pseudo label for test mode
        #    label = np.array(0, dtype=np.float32)
        extracted_image_feature = self.image_feature_dict[article_id]
        # Additional info for feeding FC layer
        flat_features = []
        sex = self.sex[gender]
        label_onehot = np.zeros(2, dtype=np.float32)
        label_onehot[sex - 1] = 1
        flat_features.extend(label_onehot)
        age = self.age[age_range]
        label_onehot = np.zeros(9, dtype=np.float32)
        label_onehot[age - 1] = 1
        flat_features.extend(label_onehot)
        time = hh
        label_onehot = np.zeros((24), dtype=np.float32)
        label_onehot[time - 1] = 1
        flat_features.extend(label_onehot)
        flat_features.append(history_num) #history number add

        flat_features.append(self.distcnts[article_id]) #base article이 base aritcle set에는 몇개나?

        try:
            flat_features.append(self.history_distcnts[article_id]) #base article이 history article에는 몇개나?
        except:
            flat_features.append(0)

        try:
            flat_features.append(self.history_distcnts[history_dupicate_top1]) #history article이 history article set에는 몇개나?
        except:
            flat_features.append(0)
        try:
            flat_features.append(self.distcnts[history_dupicate_top1]) #history article이 base set article에는 몇개나?
        except:
            flat_features.append(0)

        if history_dupicate_top1 == "NoDup":
            history_feature = np.zeros(extracted_image_feature.shape) #history article의 feature 이미지가 없음..
        else:
            history_feature = self.image_feature_dict[history_dupicate_top1] #history article의 feature
            
        flat_features = np.array(flat_features).flatten()
        flat_features =np.concatenate((flat_features, extracted_image_feature), axis=None)
        flat_features =np.concatenate((flat_features, history_feature), axis=None)
        # hint: flat features are concatened into a Tensor, because I wanted to put them all into computational model,
        # hint: if it is not what you wanted, then change the last return line
        return flat_features, label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.item.index.values.tolist()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
