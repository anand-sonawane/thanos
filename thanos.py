import argparse
from Thanos import train
from Thanos import test

"""

All the recognition models available:

keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet','mobilenet']
keras_contrib_models = ['wideresnet','ror']
other = ['resnet101','resnet152']
all_other = ['resnet18','resnet34','squeezenet1_0','squeezenet1_1','vgg11','vgg13','resnext101_64x4d','resnext101_32x4d','inceptionv4']

"""
def print_arguments():
    print('task : ',args.task)
    print('data_dir_train : ',args.data_dir_train)
    print('data_dir_valid : ',args.data_dir_valid)
    print('data_dir_test : ',args.data_dir_test)
    print('model_name : ',args.model_name)
    print('epochs : ',args.epochs)
    print('batch_size :',args.batch_size)
    print('training_type :',args.training_type)
    print('save_loc : ',args.save_loc)
    print('weights : ',args.weights)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a Image Classification model')

    required = parser.add_argument_group('required arguments')
    required.add_argument('-t', '--task',help = 'Train or Test',required = True)

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-dtrain', '--data_dir_train', default="../data/training_data", help = 'Path to Training Data')
    optional.add_argument('-dvalid', '--data_dir_valid', default="../data/validation_data", help = 'Path to Validation Data')
    optional.add_argument('-dtest', '--data_dir_test', default="../data/testing_data", help = 'Path to Testing Data')
    optional.add_argument('-m', '--model_name', default="resnet50",help = 'Pretrianed model name')
    optional.add_argument('-e', '--epochs', default=100, type=int, help = 'Number of epochs')
    optional.add_argument('-b', '--batch_size', default=32, type=int , help = 'Batch-size')
    #We will be using 3 Training Types - 1 : Fine tune all network , 2: Freeze some starting layers
    optional.add_argument('-tt', '--training_type',default = "fine_tune",help = 'Fine tune all network: fine_tune , Freeze the starting layers : freeze')
    optional.add_argument('-s', '--save_loc', default="models/" ,help = 'Save location for the trained models')
    optional.add_argument('-w', '--weights',default = 'imagenet', help='weights imagenet or custom')



    """
    #additional options for v2
    optional.add_argument('-cs', '--crop-size', type=int, default=512, help='Crop size')
    optional.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
    optional.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for FC layers')
    optional.add_argument("-p", '--use_parallel', default=False, action='store_true')
    optional.add_argument("-a", '--aug', default=False, action='store_true',help = 'Apply Ba0sic Augumentation or not' )
    optional.add_argument("-act", '--activation', default= 'relu' , help = 'Activation Function to be used')
    optional.add_argument('-o', '--optimizer',default = "sgd", help = 'The optimizer to be used')
    optional.add_argument("-g", '--no_of_gpus', default=1,help = 'No of GPUs to be used for training)
    """


    args = parser.parse_args()
    print_arguments()


    keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet','mobilenet']
    keras_contrib_models = ['wideresnet','ror']
    other = ['resnet101','resnet152']
#	else:
#        print("Please input the correct model name")


    """
    Using the argument parser inputs :
    The following files will be used :
    train.py : To train Classification models and save them to the provided location
    test.py : To test the models created after loading their weights
    """

    if args.task.lower() == 'train':
        train.train_model(args.data_dir_train,args.data_dir_valid,args.batch_size,args.epochs,args.model_name,args.training_type,args.save_loc,args.weights)

    elif args.task.lower() == 'test': print(test)

    else:
        print("Incorrect task")
