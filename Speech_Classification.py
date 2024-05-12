'''
This Python script implements the GAN (Generative Adversarial Network)-based speech recognition system designed for real-time applications in consumer electronics. 
The script demonstrates how one-dimensional speech signals are processed and recognized using a GAN model, with a focus on achieving high accuracy and low latency in speech command recognition.

The proposed system begins by capturing an audio signal through a microphone, which is then processed to distinguish between speech and ambient noise. 
Valid speech signals are transformed into a two-dimensional spectrogram that serves as the input for the GAN model. 
The GAN model consists of a Generator and a Discriminator working in tandem to classify speech signals. 
The Generator crafts fake spectrogram images from random noise, while the Discriminator learns to differentiate between these fakes and real spectrograms, effectively classifying the speech.

Paper: https://ieeexplore.ieee.org/document/10134295
'''

pip install ffmpeg-python

from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint
#import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Model, Sequential

from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation

from PIL import Image
import os
import numpy as np

from matplotlib import pyplot as plt
from keras import backend as K

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle


### Loading the Dataset
def load_custom():
  label_1 = []
  vectorized_images_x = []
  count = 0
  count1 = 0
  path_to_files_train_0 = "/content/drive/MyDrive/audio/0/"
  path_to_files_train_1 = "/content/drive/MyDrive/audio/1/"
  path_to_files_train_2 = "/content/drive/MyDrive/audio/2/"
  path_to_files_train_3 = "/content/drive/MyDrive/audio/3/"
  path_to_files_train_4 = "/content/drive/MyDrive/audio/4/"
  path_to_files_train_5 = "/content/drive/MyDrive/audio/5/"
  path_to_files_train_6 = "/content/drive/MyDrive/audio/6/"
  path_to_files_train_7 = "/content/drive/MyDrive/audio/7/"
  path_to_files_train_8 = "/content/drive/MyDrive/audio/8/"
  path_to_files_train_9 = "/content/drive/MyDrive/audio/9/"

  for img in os.listdir(path_to_files_train_0):
          #if your image name contain 'contrast'
    label_1.append((0))
    image_1 = Image.open(path_to_files_train_0 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
    #count = count+1
  for img in os.listdir(path_to_files_train_1):
    label_1.append((1))
    image_1 = Image.open(path_to_files_train_1 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_2):
    label_1.append((2))
    image_1 = Image.open(path_to_files_train_2 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_3):
    label_1.append((3))
    image_1 = Image.open(path_to_files_train_3+ img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_4):
    label_1.append((4))
    image_1 = Image.open(path_to_files_train_4 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_5):
    label_1.append((5))
    image_1 = Image.open(path_to_files_train_5 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_6):
    label_1.append((6))
    image_1 = Image.open(path_to_files_train_6 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_7):
    label_1.append((7))
    image_1 = Image.open(path_to_files_train_7 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_8):
    label_1.append((8))
    image_1 = Image.open(path_to_files_train_8 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
  for img in os.listdir(path_to_files_train_9):
    label_1.append((9))
    image_1 = Image.open(path_to_files_train_9 + img).convert('RGB')
    image_1 = image_1.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_1)
    vectorized_images_x.append(image_array)
    #count1= count1+1
  #print(count+count1)
  np.savez("/content/drive/MyDrive/mnistlikedataset_audio.npz",DataX=vectorized_images_x,trainy=label_1)

load_dadaset=load_custom()

def test_custom():
  label_2 = []
  vectorized_images_y = []
  count_1 = 0
  count_2 = 0
  path_to_files_test_0 = "/content/drive/MyDrive/audio_test/0/"
  path_to_files_test_1 = "/content/drive/MyDrive/audio_test/1/"
  path_to_files_test_2 = "/content/drive/MyDrive/audio_test/2/"
  path_to_files_test_3 = "/content/drive/MyDrive/audio_test/3/"
  path_to_files_test_4 = "/content/drive/MyDrive/audio_test/4/"
  path_to_files_test_5 = "/content/drive/MyDrive/audio_test/5/"
  path_to_files_test_6 = "/content/drive/MyDrive/audio_test/6/"
  path_to_files_test_7 = "/content/drive/MyDrive/audio_test/7/"
  path_to_files_test_8 = "/content/drive/MyDrive/audio_test/8/"
  path_to_files_test_9 = "/content/drive/MyDrive/audio_test/9/"
  #path_to_files_test_2 = "/content/drive/MyDrive/Summer_Internship/pedestrian/inria/test/ped/"
  for img in os.listdir(path_to_files_test_0):
    label_2.append((0))
    image_2 = Image.open(path_to_files_test_0 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_1):
    label_2.append((1))
    image_2 = Image.open(path_to_files_test_1 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_2):
    label_2.append((2))
    image_2 = Image.open(path_to_files_test_2 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_3):
    label_2.append((3))
    image_2 = Image.open(path_to_files_test_3 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_4):
    label_2.append((4))
    image_2 = Image.open(path_to_files_test_4 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_5):
    label_2.append((5))
    image_2 = Image.open(path_to_files_test_5 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_6):
    label_2.append((6))
    image_2 = Image.open(path_to_files_test_6 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_7):
    label_2.append((7))
    image_2 = Image.open(path_to_files_test_7 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_8):
    label_2.append((8))
    image_2 = Image.open(path_to_files_test_8 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_9):
    label_2.append((9))
    image_2 = Image.open(path_to_files_test_9 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
    #count_1 = count_1+1'''
  np.savez("/content/drive/MyDrive/mnistlikedataset_audio_test.npz",Datay=vectorized_images_y,testy=label_2)

test_custom()

def validation_custom():
  label_2 = []
  vectorized_images_y = []
  count_1 = 0
  count_2 = 0
  path_to_files_test_0 = "/content/drive/MyDrive/audio_validation/0/"
  path_to_files_test_1 = "/content/drive/MyDrive/audio_validation/1/"
  path_to_files_test_2 = "/content/drive/MyDrive/audio_validation/2/"
  path_to_files_test_3 = "/content/drive/MyDrive/audio_validation/3/"
  path_to_files_test_4 = "/content/drive/MyDrive/audio_validation/4/"
  path_to_files_test_5 = "/content/drive/MyDrive/audio_validation/5/"
  path_to_files_test_6 = "/content/drive/MyDrive/audio_validation/6/"
  path_to_files_test_7 = "/content/drive/MyDrive/audio_validation/7/"
  path_to_files_test_8 = "/content/drive/MyDrive/audio_validation/8/"
  path_to_files_test_9 = "/content/drive/MyDrive/audio_validation/9/"
  for img in os.listdir(path_to_files_test_0):
    label_2.append((0))
    image_2 = Image.open(path_to_files_test_0 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_1):
    label_2.append((1))
    image_2 = Image.open(path_to_files_test_1 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_2):
    label_2.append((2))
    image_2 = Image.open(path_to_files_test_2 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_3):
    label_2.append((3))
    image_2 = Image.open(path_to_files_test_3 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_4):
    label_2.append((4))
    image_2 = Image.open(path_to_files_test_4 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_5):
    label_2.append((5))
    image_2 = Image.open(path_to_files_test_5 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_6):
    label_2.append((6))
    image_2 = Image.open(path_to_files_test_6 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_7):
    label_2.append((7))
    image_2 = Image.open(path_to_files_test_7 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_8):
    label_2.append((8))
    image_2 = Image.open(path_to_files_test_8 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
  for img in os.listdir(path_to_files_test_9):
    label_2.append((9))
    image_2 = Image.open(path_to_files_test_9 + img).convert('RGB')
    image_2 = image_2.resize((128,128), Image.ANTIALIAS)
    image_array = np.array(image_2)
    vectorized_images_y.append(image_array)
    #count_1 = count_1+1'''
  np.savez("/content/drive/MyDrive/mnistlikedataset_audio_validation.npz",Datay=vectorized_images_y,testy=label_2)

validation_custom()


### Training, Validation and Testing the model, and saving the model

disc_loss_real=[]
disc_loss_fake=[]
ganloss=[]
supdisc_loss=[]
supdisc_acc=[]
disc_acc=[]
z=[]
ganop=[]
testacc=[]
valacc=[]
#GLOBAL VARIABLE FOR NUMBER OF CLASSES , BATCHES , EPOCHS
n_epochs = 200
n_classes = 10
n_batch = 150
latent_dim = 100
n_samples =150

def define_generator(latent_dim):

  in_lat = Input(shape=(latent_dim,))
	#Start with enough dense nodes to be reshaped and ConvTransposed to 28x28x1
  n_nodes = 1024 * 16 * 16
  X = Dense(n_nodes)(in_lat)
  X = LeakyReLU(alpha=0.2)(X)
  X = Reshape((16, 16, 1024))(X)
  X = Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(X) #14x14x128
  X = LeakyReLU(alpha=0.2)(X)

  X = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(X) #14x14x128
  X = LeakyReLU(alpha=0.2)(X)

  X = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(X) #14x14x64
  X = LeakyReLU(alpha=0.2)(X)
	# output
  out_layer = Conv2DTranspose(3, (3,3), strides=(2,2), activation='tanh',
                             padding='same')(X) #28x28x1
	# define model
  model = Model(in_lat, out_layer)
  model.summary()
  return model


def define_discriminator(in_shape=(128,128,3), n_classes=n_classes):
    in_image = Input(shape=in_shape)
    X = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Conv2D(64, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Conv2D(128, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Conv2D(512, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Flatten()(X)
    X = Dropout(0.4)(X) #Consider adding more dropout layers to minimize overfitting - remember we work with limited labeled data.
    X = Dense(n_classes)(X)
    print(X)
    model = Model(inputs=in_image, outputs=X)
    model.summary()
    return model
define_discriminator()

def define_sup_discriminator(disc):
    model=Sequential()
    model.add(disc)
    model.add(Activation('softmax'))
    #Let us use sparse categorical loss so we dont have to convert our Y to categorical
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, epsilon=1e-07),
                  loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    return model

#Define the unsupervised discriminator
#Takes the output of the supervised, just before the softmax activation.
#Then, adds a layer with calculation of sum of exponential outputs. (defined below as custom_activation)


#This custom activation layer gives a value close to 0 for smaller activations
#in the prior discriminator layer. It gives values close to 1 for large activations.
#This way it gives low activation for fake images. No need for sigmoid anymore.

# custom activation function for the unsupervised discriminator
#D(x) = Z(x) / (Z(x) + 1) where Z(x) = sum(exp(l(x))). l(x) is the output from sup discr. prior to softmax
def custom_activation(x):
    Z_x = K.sum(K.exp(x), axis=-1, keepdims=True)
    D_x = Z_x /(Z_x+1)

    return D_x

def define_unsup_discriminator(disc):
    model=Sequential()
    model.add(disc)
    #model.add(Activation('softmax'))
    model.add(Lambda(custom_activation))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-04, beta_1=0.5, epsilon = 1e-07))
    return model
# define the combined generator and discriminator model, for updating the generator
def define_gan(gen_model, disc_unsup):

  disc_unsup.trainable = True # make unsup. discriminator not trainable
  gan_output = disc_unsup(gen_model.output) #Gen. output is the input to disc.
  model = Model(gen_model.input, gan_output)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-04, beta_1=0.5, epsilon=1e-07))
  ganop.append(gan_output)
  return model

#gan_model = define_gan(gen_model, disc_unsup)
#print("GAN MODEL SUMMARY::")
#print(gan_model.summary())

# load the images
def load_real_samples(n_classes=n_classes):
  path = "/content/drive/MyDrive/mnistlikedataset_audio.npz"
  with np.load(path , allow_pickle=True) as data:
    #load DataX as train_data
    trainX = data['DataX']
    trainy = data['trainy']
    (trainX, trainy) = (trainX, trainy)
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5  # scale from [0,255] to [-1,1] as we will be using tanh activation.
    print(X.shape, trainy.shape)
    return [X, trainy]


#USED TO SELECT IMAGES. BUT IN OUR CASE ITS CHOOSING ALL THE 4 IMAGES
def select_supervised_samples(dataset, n_samples=n_samples, n_classes=n_classes):
  X, y = dataset
  X_list, y_list = list(), list()
  X_list = X
  y_list = y
  #print(X_list)
  #print(y_list)
  return asarray(X_list), asarray(y_list)


# Pick real samples from the dataset.
#Return both images and corresponding labels in addition to y=1 indicating that the images are real.
#Remember that we will not use the labels for unsupervised, only used for supervised.
def generate_real_samples(dataset, n_samples=n_samples):

	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix] #Select random images and corresponding labels
	y = ones((n_samples, 1)) #Label all images as 1 as these are real images. (for the discriminator training)
	return [X, labels], y

# generate latent points, to be used as inputs to the generator.
def generate_latent_points(latent_dim, n_samples=n_samples):
  z_input = randn(latent_dim * n_samples)
  z_input = z_input.reshape(n_samples, latent_dim) # reshape for input to the network
  #print(z_input)
  #z.append(z_input)
  return z_input

# Generate fake images using the generator and above latent points as input to it.
#We do not care about labeles so the generator will not know anything about the labels.
def generate_fake_samples(generator, latent_dim, n_samples=n_samples):

  z_input = generate_latent_points(latent_dim, n_samples)
  fake_images = generator.predict(z_input)
	# create class labels
  y = zeros((n_samples, 1)) #Label all images as 0 as these are fake images. (for the discriminator training)
  #print(fake_images)

  return fake_images, y

# report accuracy and save plots & the model periodically.
def summarize_performance(step, gen_model, disc_sup, latent_dim, dataset, n_samples=n_samples):
	# Generate fake images
  B, _ = generate_fake_samples(gen_model, latent_dim, n_samples)
  B = (B + 1) / 2.0 # scale to [0,1] for plotting
  A, _ = select_supervised_samples(dataset, n_samples=n_samples, n_classes=n_classes)
  A, _ = load_real_samples(n_classes)
  A = (A+1)/2.0


  X, y = dataset
  X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, n_batch)
  _, acc = disc_sup.evaluate(X, y, verbose=0)
  disc_acc.append(acc)
  print('Discriminator Accuracy: %.3f%%' % (acc * 100))
  #print('Unsup Disc Output')
  #print(ganop)
  #ac = disc_unsup.evaluate(X_fake , y_fake, verbose=0)
  #print('Unsup Discriminator Accuracy: %.3f%%' % (ac * 100))



# train the generator and discriminator
def train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=n_epochs, n_batch=n_batch):

    # select supervised dataset for training.

  X_sup, y_sup = select_supervised_samples(dataset)
	#print(X_sup.shape, y_sup.shape)

  bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# iterations
  n_steps = bat_per_epo * n_epochs

  #half_batch = int(n_batch / 2)
  print('n_epochs=%d, n_batch=%d, b/e=%d, steps=%d' % (n_epochs,
                                                              n_batch,
                                                              bat_per_epo, n_steps))

    #  enumerate epochs
  for j in range(n_epochs):
    for i in range(bat_per_epo):
		# update supervised discriminator (disc_sup) on real samples.
        #Remember that we use real labels to train as this is supervised.
        #This is the discriminator we really care about at the end.
        #Also, this is a multiclass classifier, not binary. Therefore, our y values
        #will be the real class labels for MNIST. (NOT 1 or 0 indicating real or fake.)
      [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], n_batch)
      sup_loss, sup_acc = disc_sup.train_on_batch(Xsup_real, ysup_real)


		# update unsupervised discriminator (disc_unsup) - just like in our regular GAN.
        #Remember that we will not train on labels as this is unsupervised, just binary as in our regular GAN.
        #The y_real below indicates 1s telling the discriminator that these images are real.
        #do not confuse this with class labels.
        #We will discard this discriminator at the end.
      [X_real, _], y_real = generate_real_samples(dataset, n_batch)
      d_loss_real  = disc_unsup.train_on_batch(X_real, y_real)
        #Now train on fake.
      X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, n_batch)
      d_loss_fake  = disc_unsup.train_on_batch(X_fake, y_fake)

		# update generator (gen) - like we do in regular GAN.
        #We can discard this model at the end as our primary goal is to train a multiclass classifier (sup. disc.)
      X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
      gan_loss = gan_model.train_on_batch(X_gan, y_gan)
      supdisc_acc.append(sup_acc)
      supdisc_loss.append(sup_loss)
      ganloss.append(gan_loss)
      disc_loss_real.append(d_loss_real)
      disc_loss_fake.append(d_loss_fake)
    # evaluate the model performance every 'epoch'
    #plot_history(a4, a5, a3, a2)
    # record history

		# summarize loss on this batch
    print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (j+1, sup_loss, sup_acc*100, d_loss_real, d_loss_fake, gan_loss))
		# evaluate the model performance periodically



		# evaluate the model performance periodically


    #if (j+1) % ((n_epochs/10) * 1) == 0:
    summarize_performance(i, gen_model, disc_sup, latent_dim, dataset)

    filename2 = 'gen_model_%04d.h5' % (j+1)
      #gen_model.save(filename2)
	# save the Discriminator (classifier) model
    filename3 = 'disc_sup_%04d.h5' % (j+1)
    disc_sup.save(filename3)

    filename4 = 'disc_unsup_%04d.h5' %(j+1)
      #disc_unsup.save(filename4)
    print('>Saved: %s, %s and %s' % ( filename2, filename3 , filename4))
    from keras.models import load_model


# load the model

    disc_sup_trained_model = load_model('disc_sup_%04d.h5' % (j+1))

# load the dataset
    path = "/content/drive/MyDrive/mnistlikedataset_audio_test.npz"
    with np.load(path) as data:
    #load DataX as train_data
      testX = data['Datay']
      testy = data['testy']
    (testX, testy) = (testX , testy)

# expand to 3d, e.g. add channels
    testX = expand_dims(testX, axis=-1)

# convert from ints to floats
    testX = testX.astype('float32')

# scale from [0,255] to [-1,1]
    testX = (testX - 127.5) / 127.5

# evaluate the model
    _, test_acc = disc_sup_trained_model.evaluate(testX, testy, verbose=0)
    print('Test Accuracy: %.3f%%' % (test_acc * 100))
    testacc.append(test_acc)
    #y_pred_test = disc_sup_trained_model.predict(testX)
#print(y_pred_test)
    #prediction_test = np.argmax(y_pred_test, axis=1)
    #print(prediction_test)
    path = "/content/drive/MyDrive/mnistlikedataset_audio_validation.npz"
    with np.load(path) as data:
    #load DataX as train_data
      tX = data['Datay']
      ty = data['testy']
    (tX, ty) = (tX , ty)

# expand to 3d, e.g. add channels
    tX = expand_dims(tX, axis=-1)

# convert from ints to floats
    tX = tX.astype('float32')

# scale from [0,255] to [-1,1]
    tX = (tX - 127.5) / 127.5

# evaluate the model
    _, val_acc = disc_sup_trained_model.evaluate(tX, ty, verbose=0)
    print('Val Accuracy: %.3f%%' % (val_acc * 100))
    valacc.append(val_acc)
latent_dim = 100
# create the discriminator models
disc=define_discriminator() #Bare discriminator model...
disc_sup=define_sup_discriminator(disc) #Supervised discriminator model
disc_unsup=define_unsup_discriminator(disc) #Unsupervised discriminator model.
#gen_model = load_model('disc_sup_1000.h5')
gen_model = define_generator(latent_dim) #Generator
gan_model = define_gan(gen_model, disc_unsup) #GAN
dataset = load_real_samples() #Define the dataset by loading real samples. (This will be a list of 2 numpy arrays, X and y)
	# save the generator model
%time l=train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs, n_batch)

#### Graphs

n1=list(range(1,n_epochs+1))
#n2=list(range(1,11))
def plot_history(supdisc_loss, supdisc_acc,ganloss,disc_loss_fake,disc_loss_real,disc_acc,n1,valacc):
  plt.figure(figsize=(10,10))
  plt.plot(n1,supdisc_loss, label='sup_loss')
  plt.legend()
  plt.savefig('./plot_sup_loss.png')
  plt.show()
  plt.figure(figsize=(10,10))
  plt.plot(n1,valacc, label='val_acc')
  plt.legend()
  plt.savefig('./plot_val_acc.png')
  plt.show()
  plt.figure(figsize=(10,10))
  plt.plot(n1,ganloss, label='gan_loss')
  plt.legend()
  plt.savefig('./plot_gan_loss.png')
  plt.show()
  plt.figure(figsize=(10,10))
  plt.plot(n1,disc_loss_real, label='d_loss_real')
  plt.legend()
  plt.savefig('./plot_d_loss_real.png')
  plt.show()
  plt.figure(figsize=(10,10))
  plt.plot(n1,disc_loss_fake, label='d_loss_fake')
  plt.legend()
  plt.savefig('./plot_d_loss_fake.png')
  plt.show()
  plt.figure(figsize=(10,10))
  plt.plot(n1,supdisc_acc, label='sup_acc')
  plt.legend()
  plt.savefig('./plot_sup_acc.png')
  plt.show()
  plt.figure(figsize=(10,10))
  plt.plot(n1,disc_acc,color='b', label='train_acc')
  #plt.plot(n2,testacc, color='r',label='test_acc')
  plt.legend()
  plt.savefig('./plot_epoch_vs_acc.png')
  plt.show()
plot_history(supdisc_loss, supdisc_acc,ganloss,disc_loss_fake,disc_loss_real,disc_acc,n1,valacc)
def epochtestacc(testacc):
  plt.figure(figsize=(10,10))
  plt.plot(testacc, label='testacc')
  plt.legend()
  plt.savefig('./testaccvsepoch')
  plt.show()
epochtestacc(testacc)

def traintestval(disc_acc, testacc):

  plt.figure(figsize=(10,10))
  plt.plot(disc_acc, label ='train_acc')
  plt.plot(valacc, label = 'val_acc')
  #plt.plot(testacc, label = 'test_acc')
  plt.xlabel('EPOCHS')
  plt.legend()
  plt.savefig('./traintest.png')
  plt.show()

traintestval(disc_acc, testacc)

