import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras import layers as l 
from keras_self_attention import SeqSelfAttention

def build_model(n_vocab):
  inputs = l.Input(shape=(100, 1))
  lstm1 = l.Bidirectional(l.LSTM(units=512, input_shape=(100, 1), return_sequences=True))(inputs)
  
  drop = l.Dropout(0.3)(lstm1)
  attn = SeqSelfAttention(attention_activation='sigmoid')(drop)
  lstm2 = l.LSTM(units=512, return_sequences=False)(attn)

  bn1 = l.BatchNormalization()(lstm2)

  flat = l.Flatten()(bn1)
  layer_out = l.Dense(n_vocab, activation='softmax')(flat)

  model = Model(inputs, layer_out)
  adam = tf.keras.optimizers.Adam(lr=0.001)
  model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

  return model

model = build_model(223)
print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
        'model_check.h5',
        preriod=10,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        mode='min'
    )
callbacks_list = [checkpoint]

model.fit(in_seq, out_seq, epochs=200, callbacks=callbacks_list)