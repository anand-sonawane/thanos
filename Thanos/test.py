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
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint

z = glob.glob('/data/Rajneesh/PnGbottleYolo/anand/plant/data/test/*.*')
test_imgs = []
names = []
for fn in z:
    names.append(fn.split('/')[-1])
    new_img = Image.open(fn)
    test_img = ImageOps.fit(new_img, (224, 224), Image.ANTIALIAS).convert('RGB')
    test_imgs.append(test_img)

#load the saved model
model = load_model('model.h5')


timgs = np.array([np.array(im) for im in test_imgs])
timgs = timgs.astype(float)
testX = timgs.reshape(timgs.shape[0], 224, 224, 3) / 255


#predict

yhat = model.predict(testX)
test_y = lb.inverse_transform(yhat)

#create submission file

df = pd.DataFrame(data={'file': names, 'species': test_y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('results.csv', index=False)
