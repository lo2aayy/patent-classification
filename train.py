import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from preprocess_data import preprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-embedding', action='store', dest='embedding')
fasttext_name = parser.parse_args().embedding

embedding_dim = 300
learning_rate = 0.001145
bs = 128
drop = 0.2584
max_length = 1431
max_num_words = 23140
filters = [6]
num_filters = 2426
nclasses = 451

x_train, y_train, x_val, y_val, embedding_matrix = preprocess(fasttext_name,
                                                              embedding_dim,
                                                              max_length,
                                                              max_num_words)

print("Starting Training ...")

filter_sizes = []
for i in filters:
    filter_sizes.append(i)

embedding_layer = Embedding(max_num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=True)

sequence_input = Input(shape=(max_length,), dtype='uint16')
embedded_sequences = embedding_layer(sequence_input)
reshape = Reshape((max_length, embedding_dim, 1))(embedded_sequences)

maxpool_blocks = []
for filter_size in filter_sizes:
    conv = Conv2D(num_filters, kernel_size=(filter_size, embedding_dim),
                  padding='valid', activation='relu',
                  kernel_initializer='he_uniform',
                  bias_initializer='zeros')(reshape)
    maxpool = MaxPool2D(pool_size=(max_length - filter_size + 1, 1),
                        strides=(1, 1), padding='valid')(conv)
    maxpool_blocks.append(maxpool)

if len(maxpool_blocks) > 1:
    concatenated_tensor = Concatenate(axis=1)(maxpool_blocks)
else:
    concatenated_tensor = maxpool_blocks[0]

flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=nclasses, activation='softmax')(dropout)

model = Model(inputs=sequence_input, outputs=output)

adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_train = to_categorical(np.asarray(y_train),
                         num_classes=nclasses).astype(np.float16)
y_val = to_categorical(np.asarray(y_val),
                       num_classes=nclasses).astype(np.float16)
history = model.fit(x_train, y_train,
                    batch_size=bs, shuffle=True,
                    epochs=20,
                    validation_data=(x_val, y_val))
