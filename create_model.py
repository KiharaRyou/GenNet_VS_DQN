from tensorflow import keras

def create_network(input_shape, action_size, layers):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=input_shape))
    if layers:
        for layer in layers:
            model.add(layer)
    model.add(keras.layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model

def create_dense_layer(n_unit):
    return keras.layers.Dense(n_unit, activation='relu')