import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Activation
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import precision_score, recall_score
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Seed for reproducibility
np.random.seed(1337)


def reshape_data(arr, img_rows, img_cols, channels):
    """
    Reshapes the data into format for CNN.
    """
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def build_cnn_model(input_shape, kernel_size, nb_filters, nb_classes):
    """
    Defines the Convolutional Neural Network model with mathematical significance explained.
    INPUT
        input_shape: Shape of the input data (image dimensions + channels)
        kernel_size: Size of the kernel for the convolution layers (e.g., 3x3, 5x5)
        nb_filters: Number of filters for the convolutional layers
        nb_classes: Number of output classes
    OUTPUT
        Compiled Keras model
    """
    model = Sequential()

    # First Conv Layer: Detects local patterns (e.g., edges)
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))  # Applies non-linearity to introduce complexity
    model.add(BatchNormalization())  # Normalizes to stabilize training

    # Second Conv Layer: Learns more complex features from the first layer
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Reduces spatial dimensions, retains essential information
    model.add(Dropout(0.25))  # Regularization to prevent overfitting

    # Third Conv Layer: Further refinement of features
    model.add(Conv2D(nb_filters * 2, kernel_size=kernel_size, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening Layer: Converts 2D feature maps into 1D vector
    model.add(Flatten())

    # Fully Connected Layer: Combines all features for final classification
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Reduces overfitting by dropping 50% of the neurons

    # Output Layer: Classification layer using softmax for multi-class output
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Compiling the model with Adam optimizer and categorical crossentropy loss
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def save_model(model, score, model_name):
    """
    Save the trained model if the evaluation score exceeds a threshold.
    """
    if score >= 0.75:  # Save if the model's recall score is above 0.75
        model.save(f'{model_name}.h5')
        logging.info(f"Model saved as {model_name}.h5 with score: {score}")
    else:
        logging.info(f"Model not saved. Score: {score}")


if __name__ == '__main__':
    # Parameters before model training
    batch_size = 512
    nb_classes = 2  # Binary classification
    nb_epoch = 30
    img_rows, img_cols = 256, 256
    channels = 3
    nb_filters = 32
    kernel_size = (3, 3)  # Standard size for convolution kernels

    # Load the preprocessed data (from numpy arrays after preprocessing)
    logging.info("Loading preprocessed data")
    X_train = np.load("../data/X_train.npy")
    X_test = np.load("../data/X_test.npy")
    y_train = np.load("../data/y_train.npy")
    y_test = np.load("../data/y_test.npy")

    # Reshape the data for CNN
    logging.info("Reshaping data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    # Normalization: Scale pixel values to the range [0, 1]
    logging.info("Normalizing data")
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # One-hot encoding of the labels (ensure labels are categorical)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # Define the CNN model
    logging.info("Building and training the CNN model")
    input_shape = (img_rows, img_cols, channels)
    model = build_cnn_model(input_shape, kernel_size, nb_filters, nb_classes)

    # Model training with early stopping and TensorBoard
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2,
              callbacks=[early_stopping, tensorboard], verbose=1)

    # Evaluation and predictions
    logging.info("Evaluating the model")
    score = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f'Test score: {score[0]}, Test accuracy: {score[1]}')

    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test_labels, y_pred_labels)
    recall = recall_score(y_test_labels, y_pred_labels)

    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")

    # Save the model if the recall score is above a threshold
    save_model(model=model, score=recall, model_name="DR_Two_Classes_CNN")
    logging.info("Training and evaluation completed.")
