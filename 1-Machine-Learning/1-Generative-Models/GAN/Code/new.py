import tensorflow as tf
from tensorflow.keras import layers

train_x = tf.random.normal([10000, 10])
train_y = tf.ones_like(train_x)



model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(10, )))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2))


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def loss_fn(generated_solve):
    out = generated_solve[:,0]**2 + generated_solve[:,1]**2
    return cross_entropy(tf.zeros_like(out), out)


model.compile(optimizer='adam',
            loss='binary_crossentropy')

history = model.fit(train_x, train_y,
                    batch_size= 64,
                    epochs= 30,
                    validation_split=0.2 #分割一部分训练数据用于验证
                   )