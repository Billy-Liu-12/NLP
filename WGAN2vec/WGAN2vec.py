from __future__ import print_function, division
# https://github.com/robert-d-schultz/gan-word-embedding

# from keras.layers import Input, Dense, Reshape, Flatten
# from keras.layers import BatchNormalization, Activation
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.models import Sequential, Model

# error occur!
# from keras.optimizers import RMSprop
# use instead
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model

import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import gensim

# Read Data
raw_data = pd.read_csv("./train.txt", sep="\n")

# ignore length <7 sentence,put it in length=7, split sentence 0:7 which length >7
data = []
for i in raw_data["text"]:
    split_text = i.split(" ")
    if len(split_text) < 7:
        pass
    elif len(split_text) == 7:
        data.append(split_text)
    else:
        data.append(split_text[0:7])

# make word2vec model, dimension = 64
model = gensim.models.Word2Vec(data, size=64)

# change sentence to vector stack
sentence_to_word_vec = np.zeros(shape=(len(data), 7, 64))
for sentence_index, i in enumerate(data):
    temp_list = np.zeros(shape=(7, 64))
    for idx, j in enumerate(i):
        try:
            temp_list[idx] = np.array([model.wv.get_vector(j)])
        except:
            temp_list[idx] = np.array([model.wv.get_vector("<unk>")])
    temp_list = np.reshape(temp_list, (1, 7, 64))
    sentence_to_word_vec[sentence_index] = temp_list

# add one dimension
sentence_to_word_vec = np.expand_dims(sentence_to_word_vec, axis=-1)

class WGAN2vec():
    def __init__(self):
        self.img_rows = 7
        self.img_cols = 64
        self.channels = 1
        self.sentence_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        sentence = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(sentence)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        # model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        # model.add(Reshape((7, 7, 128)))
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        # model.add(Activation("tanh"))

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Reshape((1, 1, 512)))
        model.add(Conv2DTranspose(256, kernel_size=(3, 16), strides=2))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(1, kernel_size=(3, 34), strides=2))
        model.add(Reshape((7, 64, 1)))

        model.summary()
        # model.save("WGAN2vec_generator.h5")

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        # model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(1))

        model.add(Conv2D(256, kernel_size=(3, 64), input_shape=self.sentence_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=(5, 1)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        # model.save("WGAN2vec_discriminator.h5")

        img = Input(shape=self.sentence_shape)
        validity = model(img)

        return Model(img, validity)

    def pretrain_D(self, epochs, batch_size=128):
        X_train = sentence_to_word_vec
        # valid = np.ones((batch_size, 1))
        valid = -np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        # fake = fake + 0.1
        print("pretraining D")
        for epoch in range(epochs):
            print("{}epochs".format(epoch))
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sentences = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_sentences = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            # d_loss_real = self.discriminator.train_on_batch(sentences, valid)
            # d_loss_fake = self.discriminator.train_on_batch(gen_sentences, fake)
            d_loss_real = self.critic.train_on_batch(sentences, valid)
            d_loss_fake = self.critic.train_on_batch(gen_sentences, fake)
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = sentence_to_word_vec

        # Adversarial ground truths
        # valid = np.ones((batch_size, 1))
        valid = -np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        valid = valid * 0.9
        # fake = fake + 0.1
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sentences = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_sentences = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            # d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss_real = self.critic.train_on_batch(sentences, valid)
            d_loss_fake = self.critic.train_on_batch(gen_sentences, fake)
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # Clip critic weights
            for l in self.critic.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.show_sentence(epoch)

    def show_sentence(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = self.generator.predict(noise)
        test = np.squeeze(gen_sentence)
        for i in test:
            sentence = ""
            for j in i:
                temp = model.wv.similar_by_vector(j)
                sentence = sentence + temp[0][0] + " "
            print(sentence)

    def predict(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_sentence = self.generator.predict(noise)
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
    wgan2vec = WGAN2vec()
    wgan2vec.pretrain_D(epochs=100)
    wgan2vec.train(epochs=4000, batch_size=32, sample_interval=50)
    for i in range(100):
        t = wgan2vec.predict()
        print(t)