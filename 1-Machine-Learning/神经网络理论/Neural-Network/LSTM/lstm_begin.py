from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

import tensorflow as tf
from tensorflow.keras import layers

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# define class labels
labels = [1,1,1,1,1,0,0,0,0,0]

# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

# # pad documents to a max length of 4 words
# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)

# # define the model
# model = tf.keras.models.Sequential()
# model.add(layers.Embedding(vocab_size, 8, input_length=max_length))
# model.add(layers.Flatten())
# model.add(layers.Dense(1, activation='sigmoid'))

# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# # summarize the model
# print(model.summary())

# # fit the model
# model.fit(padded_docs, labels, epochs=50, verbose=0)

# # evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))