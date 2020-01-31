import tensorflow as tf

# importando uma base de dados
mnist = tf.keras.datasets.mnist

# separando os dados e tratando
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# criando a rede neural
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# gerando a rede neural
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# treinamento da rede neural
model.fit(x_train, y_train, epochs=15)

# avaliacao da performance
print(model.evaluate(x_test, y_test))

