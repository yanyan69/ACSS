import tensorflow as tf

def save_model(model, path='copra_classifier/models/copra_model.keras'):
    model.save(path)
    print(f'Model saved to:{path}')

def load_model(path='copra_classifier/models/copra_model.keras'):
    model = tf.keras.models.load_model(path)
    print(f'Model loaded from: {path}')
    return model