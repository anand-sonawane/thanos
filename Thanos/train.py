# Import keras Libraries
from time import time
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# Import other python Libraries
import os, math

#Import user-defined Libraries
from Thanos.models import Models

""" Code for using fraction of the GPU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

"""

def train_model(data_dir_train,data_dir_valid,batch_size,epochs,model_name,training_type,save_loc,weights):

    DATA_TRAIN = data_dir_train
    DATA_VALID = data_dir_valid
    BATCH_SIZE = batch_size
    EPOCH = epochs
    num_classes = len(os.listdir(DATA_TRAIN))

    nb_train_samples = sum([len(files) for r, d, files in os.walk(DATA_TRAIN)])
    nb_validation_samples = sum([len(files) for r, d, files in os.walk(DATA_VALID)])

    print("Training Samples :",nb_train_samples)
    print("Validation Samples :",nb_validation_samples)
    print("Number of Classes :",num_classes)

    #differenent sources from where the models are being initialized
    keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet121','densenet169','densenet201','mobilenet']
    #keras_contrib_models = ['wideresnet','ror']
    #other = ['resnet101','resnet152']

    modelsBuilder = Models(model_name,training_type,num_classes)

    if model_name in keras_models:
        model_final,img_width,img_height = modelsBuilder.createModelBase()

    """
    This feature is under development

    elif model_name in keras_contrib_models:
        model_final,img_width,img_height = models_keras_contrib.create_model(model_name,training_type,num_classes)
    else:
        model_final,img_width,img_height = models_other.create_model(model_name,training_type,num_classes)
    """

    #summary of the model built
    model_final.summary()

    # Compile the model
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # Initiate the train and test generators with data Augumentation
    train_datagen = ImageDataGenerator(
    	rescale = 1./255)

    test_datagen = ImageDataGenerator(
    	rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
    	DATA_TRAIN,
    	target_size = (img_height, img_width),
    	batch_size = BATCH_SIZE,
    	class_mode = "categorical")

    validation_generator = test_datagen.flow_from_directory(
    	DATA_VALID,
        batch_size = BATCH_SIZE,
    	target_size = (img_height, img_width),
    	class_mode = "categorical")

    # Save the model according to the conditions
    model_save = save_loc + model_name +" weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(model_save, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # ================================================================== #
    # learning rate schedule
    def step_decay(EPOCH):
    	initial_lrate = 0.001
    	drop = 0.1
    	epochs_drop = 20.0
    	lrate = initial_lrate * math.pow(drop, math.floor((1+EPOCH)/epochs_drop))
    	print("==== Epoch: {0:} and Learning Rate: {1:} ====".format(EPOCH, lrate))
    	return lrate

    change_lr = LearningRateScheduler(step_decay)
    # ================================================================== #

    # Train the model
    model_final.fit_generator(
    	train_generator,
    	samples_per_epoch = nb_train_samples,
    	epochs = EPOCH,
    	validation_data = validation_generator,
    	validation_steps = nb_validation_samples,
    	callbacks = [checkpoint, early_stopping, change_lr,tensorboard])

    # Save the final model on the disk
    model_final_name = save_loc + model_name +" weights-final.hdf5"
    model_final.save(model_final_name)

    return_string = "Model saved at : "+ model_final_name

    return return_string
