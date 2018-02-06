# Import keras Libraries
from keras_contrib import applications
from keras_contrib import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# Import other python Libraries
import os, math


"""
    keras_contrib_models = ['wideresnet','ror']
    This file will train the above models available in the keras_contrib library.
    The main function in the file is create_model function which will return the model architecture and
    the image_width and image_height in the model.
"""

def create_model(model_name,training_type,num_classes):
    #Initializing the image width and image height : This will be updated as per the model which is going to be used
    img_width,img_height = 224,224

#------------------------------------------------------WideResidualNetwork-----------------------------------------------------------------------------
    if (model_name.lower() == 'wideresnet'):
        img_width,img_height = 224,224
        if(training_type == 'freeze'):
            model = applications.wide_resnet.WideResidualNetwork(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
            model.layers.pop()
            for layer in model.layers[:37]:
            	layer.trainable = False
            for layer in model.layers[37:]:
                layer.trainable = True
        else:
            model = applications.wide_resnet.WideResidualNetwork(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
            model.layers.pop()
            for layer in model.layers:
            	layer.trainable = False
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(512, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(num_classes, activation='softmax'))
        #Final model
        model_final = Model(inputs = model.input, outputs = top_model(model.output))
        """
        This can change the code to some extent but needs to be verified
        https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
        """
#------------------------------------------------------ResidualOfResidual-----------------------------------------------------------------------------
    elif (model_name.lower() == 'ror'):
            img_width,img_height = 299,299
            if(training_type == 'freeze'):
                model = applications.ror.ResidualOfResidual(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.ror.ResidualOfResidual(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers:
                	layer.trainable = False
            top_model = Sequential()
            top_model.add(Flatten(input_shape=model.output_shape[1:]))
            top_model.add(Dense(1024, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))
            #Final model
            model_final = Model(inputs = model.input, outputs = top_model(model.output))

    return model_final,img_width,img_height

def test_model(model_name,save_loc,test_generator):
        #load the saved model
        model_loc  = save_loc + model_name + ".h5"
        model = load_model(model_loc)

        #predict
        logits = model.predict_generator(test_generator,verbose = 1)

        return logits
