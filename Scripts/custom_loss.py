import tensorflow as tf

def custom_loss(factor = 1e4):
    def loss(y_true, y_pred):
        return tf.keras.losses.MSE(y_true, y_pred)/factor
    return loss
