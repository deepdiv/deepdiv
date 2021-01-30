from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.utils import np_utils
import keras
import numpy as np

# 复制模型层
def model_copy(model):
    original_layers = [l for l in model.layers]
    new_model = keras.models.clone_model(model)
    for index, layer in enumerate(new_model.layers):
        original_layer = original_layers[index]
        original_weights = original_layer.get_weights()
        layer.set_weights(original_weights)
    return new_model


# 训练模型
def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=15, verbose=True):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


# 混洗数据
def shuffle_data(X, Y, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    return X, Y
