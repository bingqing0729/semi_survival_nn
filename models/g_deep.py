import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class g_D:
    def __init__(self,config={}):
        self.config = config

    def train(self,x_train,z_train,dt_train,fail_train,ind_train,placeholder_train,x_valid,z_valid,dt_valid,fail_valid,ind_valid,placeholder_valid,beta):
        
        
        g_inputs = keras.Input(shape=(None,3), name="vs",dtype=np.float32)
        z = keras.Input(shape=(None,1), name="z",dtype=np.float32)
        dt = keras.Input(shape=(None,1),name="dt",dtype=np.float32)
        fail = keras.Input(shape=(None,1),name="fail",dtype=np.float32)
        ind = keras.Input(shape=(None,1),name="ind",dtype=np.float32)
        placeholder = keras.Input(shape=(None,1),name="p",dtype=np.float32)

        x = layers.Dense(self.config['hidden_layers_nodes'], activation=self.config['activation'])(g_inputs)
        x = layers.Dense(self.config['hidden_layers_nodes'], activation=self.config['activation'])(x)
        g_outputs = layers.Dense(1, name="g_outputs")(x)
        g_cumsum = tf.cumsum(tf.multiply(g_outputs,dt),axis=1)
        beta_z = beta[1] * z + beta[0]
        self.model = keras.Model(inputs=((g_inputs,z,dt,ind,fail),placeholder), outputs=g_cumsum)

        l = tf.divide(tf.reduce_sum(tf.multiply(tf.multiply(tf.exp(g_cumsum + beta_z),dt)-tf.multiply(g_cumsum + beta_z,fail),ind)),self.config['batch_size'])
        self.model.add_loss(l)

        optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate'],clipvalue=1.0)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config['patience'],restore_best_weights=True)
        self.model.compile(optimizer=optimizer)
        history = self.model.fit((x_train,z_train,dt_train,ind_train,fail_train), placeholder_train, validation_data=((x_valid,z_valid,dt_valid,ind_valid,fail_valid),placeholder_valid),\
            batch_size=self.config['batch_size'], epochs=2000,callbacks=[callback],verbose=2)
        

        return min(history.history['val_loss'])