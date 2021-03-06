### Thanos

Thanos was born on Saturn's moon Titan, and is the child of Eternals Mentor and Sui-San.During his teenage,Thanos had become fascinated with nihilism and death, worshipping and eventually falling in love with the physical embodiment of death, Mistress Death.As an adult, Thanos augmented his physical strength and powers mystically and **artificially**.

So,Wishing to impress Mistress Death, Thanos gathers an army of villainous aliens and begins a nuclear bombardment of Titan that kills millions of his race.Seeking universal power in the form of the Cosmic Cube, Thanos travels to Earth. But ends up getting defeated.Later on, Thanos is eventually resurrected,and collects the Infinity Gems once again.He uses the gems to create the Infinity Gauntlet, making himself omnipotent, and erases half the living things in the universe to prove his love to Death.

![alt text](https://static.comicvine.com/uploads/original/4/40015/1341796-picture_1.png)

So, just like the **Infinity stones** thanos gathers and use them to show his love towards the Mistress of Death,
**Lets gather all the super powers in Computer Vision together,as if they are the Infinity stones.**

This Repo is basically a wrapper on top of Keras which will let you build basic Computer Vision models with one line on the terminal, the steps to do, models available and instructions are mentioned below.

### To train follow the instructions below

<br>
Run thanos.py and change the parameters as mentioned :
<br>

<br>
1.Mandatory paramters --task : Define whether you want to train or test the model, specify --task train for training and --task test for testing
<br>
<br>
2.--data_dir_train : Location to the training data folder<br>
<br>
3.--data_dir_valid : Location to the validation data folder<br>
<br>

4.--data_dir_test : Location to the validation data folder<br>
you can also create a default structure like : <br>
for train and valid : data/training_data/classes and data/validation_data/classes in the folder where you have the thanos folder
for test : data/testing_data/test/images in the folder where you have the thanos folder
<br>

<br>
5.--model_name : Name of the model you need to train <br> : default is resnet50
Current available : ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet121','densenet169','densenet201','mobilenet'] 

<br>
<br>
Future release : ['wideresnet','ror','resnet101','resnet152','resnet18','resnet34','squeezenet1_0','squeezenet1_1','vgg11','vgg13','resnext101_64x4d','resnext101_32x4d','inceptionv4',etc] 
<br>

<br>
6.--epochs : Total epochs to train on : default is 100 <br>
<br>
7.--batch_size : Batch size : default is 32 <br>
<br>
8.--training_type : There are 3 training types as follows ,
8.--save_loc : Location to save the trained models : default is models/
<br>
9.--weights : which pre-trained weights to use, for now the options are imagenet or None, will be adding to these later : default is imagenet
10.--results_loc : Location to save the results after classification : default is results/


### So to train the model you will have to run the following command:

```
python thanos.py --task train --model_name resnet50 --training_type train_all

```
which will train your resnet50 model, by updating all the weights of all the layers

