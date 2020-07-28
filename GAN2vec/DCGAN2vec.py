from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import gensim
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras.models import load_model
from keras import backend as K
from keras import objectives
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce
import tensorflow as tf

# from tensorflow.contrib.distributions import RelaxedOneHotCategorical as gumbel

# import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1.contrib.distributions import RelaxedOneHotCategorical as gumbel
# import tensorflow_probability as tfp

# Read Data
raw_data = pd.read_csv("./train.txt", sep="\n")

# ignore length <7 sentence,put it in length=7, split sentence 0:7 which length >7

## train.txt에서 데이터를 추출, length가 7이하면 버림, 7이상이면 [0:7] 까지만 data를 append함
data = []
for i in raw_data["text"]:
    split_text = i.split(" ")
    if len(split_text) < 15:
        pass
    elif len(split_text) == 15:
        data.append(split_text)
    else:
        data.append(split_text[0:15])

# make word2vec model, dimension = 64
model = gensim.models.Word2Vec(data, size=64)

# change sentence to vector stack

## word vector를 row 7, col 64의 vector로 생성
sentence_to_word_vec = np.zeros(shape=(len(data), 15, 64))
for sentence_index, i in enumerate(data):
    temp_list = np.zeros(shape=(15, 64))
    for idx, j in enumerate(i):
        try:
            temp_list[idx] = np.array([model.wv.get_vector(j)])
        except:
            temp_list[idx] = np.array([model.wv.get_vector("<unk>")])
    temp_list = np.reshape(temp_list, (1, 15, 64))
    sentence_to_word_vec[sentence_index] = temp_list

# add one dimension
sentence_to_word_vec = np.expand_dims(sentence_to_word_vec, axis=-1)


class DCGAN2vec():
    def __init__(self):
        # Input shape
        self.sentence_length = 15
        self.word_dimension = 64
        self.channels = 1
        self.sentence_shape = (self.sentence_length, self.word_dimension, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        sentence = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(sentence)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Conv2DTranspose(1, kernel_size=(2, 51), strides=1))
        model.add(Reshape((15, 64, 1)))

        model.summary()
        model.save("dcgan2vec_generator.h5")
        # tf.keras.utils.plot_model(model, to_file='dcgan_generator.png', show_shapes=True, dpi=1024)
        noise = Input(shape=(self.latent_dim,))
        sentence = model(noise)

        return Model(noise, sentence)

    # @property
    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.sentence_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        model.save("dcgan2vec_discriminator.h5")
        # tf.keras.utils.plot_model(model, to_file='dcgan_discriminator.png', show_shapes=True,dpi=1024)
        sentence = Input(shape=self.sentence_shape)
        validity = model(sentence)

        return Model(sentence, validity)

    def pretrain_D(self, epochs, batch_size=128):
        X_train = sentence_to_word_vec
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        fake = fake + 0.1
        # fake = fake + np.random.rand(batch_size,1)

        print("pretraining D")
        for epoch in range(epochs):
            # print("{}epochs".format(epoch))
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sentences = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_sentences = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(sentences, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_sentences, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            if epoch % 100 == 0:
                print("pretraining D {}epoch".format(epoch))



    def train(self, epochs, batch_size=128, save_interval=500):

        # Load the dataset
        X_train = sentence_to_word_vec

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        fake = fake + 0.1
        # fake = fake + np.random.rand(batch_size,1)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.show_sentence(epoch)

    ''' DCGAN test'''
    # def train(self, epochs, batch_size=128, save_interval=50):
    #
    #     # Load the dataset
    #     (X_train, _), (_, _) = mnist.load_data()
    #
    #     # Rescale -1 to 1
    #     X_train = X_train / 127.5 - 1.
    #     X_train = np.expand_dims(X_train, axis=3)
    #
    #     # Adversarial ground truths
    #     valid = np.ones((batch_size, 1))
    #     fake = np.zeros((batch_size, 1))
    #
    #     for epoch in range(epochs):
    #
    #         # ---------------------
    #         #  Train Discriminator
    #         # ---------------------
    #
    #         # Select a random half of images
    #         idx = np.random.randint(0, X_train.shape[0], batch_size)
    #         imgs = X_train[idx]
    #
    #         # Sample noise and generate a batch of new images
    #         noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
    #         gen_imgs = self.generator.predict(noise)
    #
    #         # Train the discriminator (real classified as ones and generated as zeros)
    #         d_loss_real = self.discriminator.train_on_batch(imgs, valid)
    #         d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
    #         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    #
    #         # ---------------------
    #         #  Train Generator
    #         # ---------------------
    #
    #         # Train the generator (wants discriminator to mistake images as real)
    #         g_loss = self.combined.train_on_batch(noise, valid)
    #         # Plot the progress
    #         print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
    #
    #         # If at save interval => save generated image samples
    #         if epoch % save_interval == 0:
    #             self.save_imgs(epoch)

    def show_sentence(self, epoch):
        r, c = 15, 15
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = DCGAN2vec.generator.predict(noise)
        test = np.squeeze(gen_sentence)
        for i in test:
            sentence = ""
            for j in i:
                temp = model.wv.similar_by_vector(j)
                sentence = sentence + temp[0][0] + " "
            print(sentence)

    def predict(self):
        r, c = 15, 15
        # np.random.normal(정규분포 평균, 표준편차, (행, 열) or 개수) : 정규 분포 확률 밀도에서 표본 추출
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = DCGAN2vec.generator.predict(noise)
        test = np.squeeze(gen_sentence)
        sentence_list = []
        for i in test:
            sentence = ""
            for j in i:
                temp = model.wv.similar_by_vector(j)
                sentence = sentence + temp[0][0] + " "
            sentence_list.append(sentence)
        return sentence


if __name__ == '__main__':
    DCGAN2vec = DCGAN2vec()
    DCGAN2vec.pretrain_D(epochs=1000)
    DCGAN2vec.train(epochs=4000, batch_size=32, save_interval=50)

    for i in range(100):
        t = DCGAN2vec.predict()
        print(t)
