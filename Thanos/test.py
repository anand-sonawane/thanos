#load the required libararies

import glob
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load the keras libraries

from keras.layers import Dropout, Input, Dense, Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint

#Import user-defined Libraries
from Thanos import models_keras
from Thanos import models_keras_contrib

def test_model(data_dir_train,data_dir_test,batch_size,model_name,save_loc,results_loc):

    DATA_TRAIN = data_dir_train
    DATA_TEST = data_dir_test
    BATCH_SIZE = batch_size
    num_classes = len(os.listdir(DATA_TRAIN))

    nb_test_samples = sum([len(files) for r, d, files in os.walk(DATA_TEST)])

    print("Test Samples :",nb_test_samples)
    print("Number of Classes :",num_classes)

    #differenent sources from where the models are being initialized
    keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet121','densenet169','densenet201','mobilenet']
    keras_contrib_models = ['wideresnet','ror']
    other = ['resnet101','resnet152']

    twotwofour_models = ['resnet50','vgg19','vgg16','densenet121','densenet169','densenet201','mobilenet','wideresnet','resnet101','resnet152']
    twoninenine_models = ['inceptionv3','xception','inceptionresnetv2','ror']
    threethreeone_models = ['nasnet']

    img_width,img_height = 224,224
    if model_name in twotwofour_models:
        img_width,img_height = 224,224
    elif model_name in twoninenine_models:
        img_width,img_height = 299,299
    elif model_name in threethreeone_models:
        img_width,img_height = 331,331

    test_datagen = ImageDataGenerator(
    	rescale = 1./255)

    test_generator = test_datagen.flow_from_directory(
    	DATA_TEST,
    	target_size = (img_height, img_width),
    	class_mode = "categorical")

    if model_name in keras_models:
        logits = models_keras.test_model(model_name,save_loc,test_generator)
    elif model_name in keras_contrib_models:
        logits = models_keras_contrib.test_model(model_name,save_loc,test_generator)
    else:
        logits = models_other.test_model(model_name,save_loc,test_generator)

    logits_pd = pd.DataFrame(logits)
    save_results = results_loc + model_name + "_results.csv"
    logits_pd.to_csv(save_results,index=False)

    return_string = "Result saved at : " + save_results
    return return_string

# z = glob.glob('/data/Rajneesh/PnGbottleYolo/anand/plant/data/test/*.*')
# test_imgs = []
# names = []
# for fn in z:
#     names.append(fn.split('/')[-1])
#     new_img = Image.open(fn)
#     test_img = ImageOps.fit(new_img, (224, 224), Image.ANTIALIAS).convert('RGB')
#     test_imgs.append(test_img)

# #load the saved model
# model = load_model('model.h5')


# timgs = np.array([np.array(im) for im in test_imgs])
# timgs = timgs.astype(float)
# testX = timgs.reshape(timgs.shape[0], 224, 224, 3) / 255


# #predict

# yhat = model.predict(testX)
# test_y = lb.inverse_transform(yhat)

# #create submission file

# df = pd.DataFrame(data={'file': names, 'species': test_y})
# df_sort = df.sort_values(by=['file'])
# df_sort.to_csv('results.csv', index=False)
