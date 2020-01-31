import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras import regularizers
from Handler import Handler


class Train():
    def __init__(self):
        self.model = tf.keras.models.Sequential() # cria uma rede neural
        self.droput = 0.2
        self.reg_l1 = 0.01
        self.reg_l2 = 0.01

    def train(self, X, Y):
        #print(X.shape)
        #print(Y.shape)

        # camada convolucional
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer='random_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer='random_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='random_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        """""
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l1(self.reg_l1),
                     activity_regularizer=regularizers.l2(self.reg_l2),
                     bias_regularizer=regularizers.l1_l2(self.reg_l1, self.reg_l2)))
        self.model.add(tf.keras.layers.Dropout(self.droput))
        self.model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='random_uniform',
                     kernel_regularizer=regularizers.l1(self.reg_l1),
                     activity_regularizer=regularizers.l2(self.reg_l2),
                     bias_regularizer=regularizers.l1_l2(self.reg_l1, self.reg_l2)))
        self.model.add(tf.keras.layers.Dropout(self.droput))
        self.model.add(tf.keras.layers.Dense(units=Handler().img_size, activation=tf.nn.sigmoid))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])




        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(50, 50, 1)))
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(50, 50, 1)))

        # adicionando camadas à rede neural
        self.model.add(tf.keras.layers.Flatten()) # camada que transforma imagens em vetores
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))
        """""

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        self.model.fit(X, Y, epochs=20)
        #return self.model

    def trainU(self, X, Y):
        # camada convolucional
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),


            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),


            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),

            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),

            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),

            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),

            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Concatenate(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),

            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                   input_shape=(Handler().img_size, Handler().img_size, 1)),

            tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        ])

        self.model.fit(X, Y)
        #self.model.predict(X, verbose=1)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X, Y, epochs=20)

    def evaluate(self, X, Y):
        val_loss, val_acc = self.model.evaluate(X, Y)

        print(val_loss, val_acc)

        return val_loss, val_acc

