import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import classification_report
import time
import numpy as np

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback for logging training metrics.

    Args:
        dir_name (str): Directory name to save TensorBoard logs.
        experiment_name (str): Name of the experiment for the logs.

    Returns:
        tensorboard_callback (tf.keras.callbacks.TensorBoard): Configured TensorBoard callback.
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

def plot_loss_curves(history):
    """
    Plots training and validation loss and accuracy curves.

    Args:
        history (tf.keras.callbacks.History): History object from model training.

    Returns:
        None
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.figure()
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def get_callbacks():
    """
    Creates learning rate scheduler and early stopping callbacks.

    Returns:
        lr_scheduler (tf.keras.callbacks.ReduceLROnPlateau): Learning rate scheduler callback.
        early_stopping (tf.keras.callbacks.EarlyStopping): Early stopping callback.
    """
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    return lr_scheduler, early_stopping


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test data,
    and calculates inference time.

    Args:
        model (tf.keras.Model): Trained Keras model.
        X_test (np.ndarray): Test data.
        y_test (np.ndarray): True labels for test data.

    Returns:
        accuracy (float): Test accuracy.
        total_inference_time (float): Total inference time Over sample.
        y_pred_classes (list): List of predicted Classes 
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    # print(f'Test Accuracy: {accuracy * 100:.2f}%')

    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    total_inference_time = end_time - start_time
    # avg_inference_time = total_inference_time / len(X_test)
    # print(f'Average Inference Time per Sample: {total_inference_time:.6f} seconds')

    y_pred_classes = np.argmax(y_pred, axis=1)
    # print(classification_report(y_test, y_pred_classes))

    return accuracy, y_pred_classes, total_inference_time