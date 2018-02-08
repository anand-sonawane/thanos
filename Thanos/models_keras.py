# Import keras Libraries
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model,load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# Import other python Libraries
import os, math


"""
    keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet121','densenet169','densenet201','mobilenet']
    This file will train the above models available in the base keras library.
    The main function in the file is create_model function which will return the model architecture and
    the image_width and image_height in the model.
"""


def create_model(model_name,training_type,num_classes):
    #Initializing the image width and image height : This will be updated as per the model which is going to be used
    img_width,img_height = 224,224

#------------------------------------------------------ResNet50-----------------------------------------------------------------------------
    if (model_name.lower() == 'resnet50'):
        img_width,img_height = 224,224
        if(training_type == 'freeze'):
            model = applications.resnet50.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
            model.layers.pop()
            for layer in model.layers[:37]:
            	layer.trainable = False
            for layer in model.layers[37:]:
                layer.trainable = True
        else:
            model = applications.resnet50.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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
#------------------------------------------------------Xception-----------------------------------------------------------------------------
    elif (model_name.lower() == 'xception'):
        img_width,img_height = 299,299
        if(training_type == 'freeze'):
            model = applications.xception.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
            model.layers.pop()
            for layer in model.layers[:37]:
            	layer.trainable = False
            for layer in model.layers[37:]:
                layer.trainable = True
        else:
            model = applications.xception.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------VGG16-----------------------------------------------------------------------------

    elif (model_name.lower() == 'vgg16'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------VGG19-----------------------------------------------------------------------------

    elif (model_name.lower() == 'vgg19'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.vgg19.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.vgg19.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------VGG16-----------------------------------------------------------------------------

    elif (model_name.lower() == 'vgg16'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------InceptionV3-----------------------------------------------------------------------------

    elif (model_name.lower() == 'inceptionv3'):
            img_width,img_height = 299,299
            if(training_type == 'freeze'):
                model = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------InceptionResNetV2-----------------------------------------------------------------------------

    elif (model_name.lower() == 'inceptionresnetv2'):
            img_width,img_height = 299,299
            if(training_type == 'freeze'):
                model = applications.inception_resnet_v2.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.inception_resnet_v2.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------NASNetLarge-----------------------------------------------------------------------------

    elif (model_name.lower() == 'nasnet'):
            img_width,img_height = 331,331
            if(training_type == 'freeze'):
                model = applications.nasnet.NASNetLarge(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.nasnet.NASNetLarge(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------DenseNet121-----------------------------------------------------------------------------

    elif (model_name.lower() == 'densenet121'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.densenet.DenseNet121(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.densenet.DenseNet121(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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
#------------------------------------------------------DenseNet169-----------------------------------------------------------------------------

    elif (model_name.lower() == 'densenet169'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.densenet.DenseNet169(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.densenet.DenseNet169(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------DenseNet201-----------------------------------------------------------------------------

    elif (model_name.lower() == 'densenet201'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.densenet.DenseNet201(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.densenet.DenseNet201(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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

#------------------------------------------------------MobileNet-----------------------------------------------------------------------------

    elif (model_name.lower() == 'mobilenet'):
            img_width,img_height = 224,224
            if(training_type == 'freeze'):
                model = applications.mobilenet.MobileNet(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                model.layers.pop()
                for layer in model.layers[:37]:
                	layer.trainable = False
                for layer in model.layers[37:]:
                    layer.trainable = True
            else:
                model = applications.mobilenet.MobileNet(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
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
