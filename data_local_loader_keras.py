import pickle
import os
from PIL import Image
import numpy as np
import pandas as pd
import nsml
from utils import get_transforms
from utils import default_loader
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
def build_cnn_model(backbone= ResNet50, input_shape =  (224,224,3), use_imagenet = 'imagenet', base_freeze=True):
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

def merge_list(image_list):
    result = Counter(image_list)
    result = sorted(result.items(), key=operator.itemgetter(1),reverse=False)
    print('merege_list')
    return result

def extract_feature(model, image_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    return feature

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
        features[article_id] = cnn_feature
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
    def __init__(self,
                 root_dir,  item, label=None,
                 transform=None,shuffle=False,batch_size=200,mode='train' ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label = label
        self.item = item
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.hist_maxuse_num = 3
        self.features_model = build_cnn_model()
        self.on_epoch_end()

        full_list = self.item['article_id'].values.tolist()
        self.image_feature_dict, self.distcnts = make_features_and_distcnt(self.features_model, full_list, 'features.pkl', 'distr_cnt.pkl')
        print('count history')
        history_num = []
        log_num = 10000*10
        history_sel_num = 1
        top_history1 = []
        total_list_article=[]
        for idx in range(self.item.shape[0]):
            if(idx%log_num==0):
                print('count process',idx, '/',self.item.shape[0])
            cur_article = self.item['read_article_ids'].loc[idx]
            if type(cur_article) == str:
                list_article = cur_article.split(',')
                total_list_article.extend(list_article)
                top_history1.append(list_article[0])
            else:
                list_article = []
                top_history1.append("")

            history_num.append(len(list_article))
            #get_item_count_max(list_article)
        self.history_feature_dict, self.history_distcnts = make_features_and_distcnt(self.features_model, total_list_article, 'history_features.pkl', 'history_distr_cnt.pkl')
        self.item['history_num'] = pd.Series(history_num, index=self.item.index)
        print('self.item print')
        for c in self.item.columns:
            print(c)
            print(self.item[c].head(10))

    def __len__(self):
        return len(self.item)

    def get_hist_features(cur_article=None, sel_shuffle=False, out_shape=2048):
        if type(cur_article) == str:
            list_article = cur_article.split(',')
        else:
            list_article = []
    
        sel_number=self.hist_maxuse_num
        if sel_number > len(list_article):
            sel_number = len(list_article)

        if sel_shuffle==True:
            sel_article = random.choices(list_article, k=sel_number)
        else:
            sel_article = list_article[:sel_number]
        hist_features = []

        for idx in range(self.hist_maxuse_num):
            if idx<sel_number:
                cnn_feature = self.history_feature_dict[sel_article[idx]]
                hist_features.append(cnn_feature)
            else:
                hist_features.append(np.zeros(out_shape))

        mer_hist_np = np.array(hist_features)
        return mer_hist_np.flatten()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(idxs)
        return X, y

    def __data_generation(self, idxs):
        X = np.empty((self.batch_size, 2048))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, idx in enumerate(idxs):
            # Store sample
            X[i,] ,  y[i] = self.get_one_data(idx)

    
    def get_one_data(self, idx):
        article_id, hh, gender, age_range, read_article_ids,history_num  = self.item.loc[idx
                       , ['article_id', 'hh', 'gender', 'age_range', 'read_article_ids','history_num']]
        if self.mode== 'train':
            label = self.label.loc[idx, ['label']]
            label = np.array(label, dtype=np.float32)
        else:
            # pseudo label for test mode
            label = np.array(0, dtype=np.float32)
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
        mer_hist_np = self.get_hist_features(cur_article=read_article_ids, sel_shuffle=self.shuffle, out_shape=extracted_image_feature.shape)
        flat_features.extend(extracted_image_feature)
        flat_features.extend(mer_hist_np)
        flat_features = np.array(flat_features).flatten()
        # hint: flat features are concatened into a Tensor, because I wanted to put them all into computational model,
        # hint: if it is not what you wanted, then change the last return line
        return flat_features, label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.item.index.values.tolist()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
