import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3, Xception, DenseNet121, DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Image size and classes
img_rows, img_cols = 256, 256
channels = 3  # RGB images
classes = 5  # Number of output classes: Normal, Mild, Moderate, Severe, PDR


# Load the preprocessed data saved as .npy files
def load_data():
    """
    Load the preprocessed data from NumPy files.
    OUTPUT:
        X_train, X_test, y_train, y_test
    """
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')

    # Normalize image data to 0-1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, X_test, y_train, y_test


def build_hypermodel(hp, base_model_name):
    """
    Build a fine-tuned model with hyperparameters using the chosen pre-trained base model.
    INPUT:
        hp: Hyperparameter object from Keras Tuner
        base_model_name: The name of the base model
    OUTPUT:
        A compiled model with the chosen base model and tuned hyperparameters.
    """
    if base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
    elif base_model_name == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
    elif base_model_name == 'DenseNet169':
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))

    # Add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Tune the number of units in the Dense layer
    units = hp.Int('units', min_value=512, max_value=2048, step=256)
    x = Dense(units, activation='relu')(x)

    # Tune the dropout rate
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    x = Dropout(dropout_rate)(x)

    # Output layer
    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Tune the learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_pretrained_models():
    """
    Load and fine-tune the ensemble of pre-trained models using hyperparameter tuning.
    OUTPUT:
        A list of fine-tuned models with the best hyperparameters.
    """
    # List of pre-trained model names
    base_models = ['ResNet50', 'InceptionV3', 'Xception', 'DenseNet121', 'DenseNet169']

    # Initialize an empty list for the best models
    best_models = []

    # Iterate over the base models and tune each one
    for base_model_name in base_models:
        print(f"Hyperparameter tuning for {base_model_name}...")

        # Initialize the Keras Tuner for hyperparameter tuning
        tuner = kt.RandomSearch(
            lambda hp: build_hypermodel(hp, base_model_name),
            objective='val_accuracy',  # Tune for validation accuracy
            max_trials=5,  # Number of different hyperparameter combinations to try
            executions_per_trial=2,  # Run each trial twice and average results
            directory=f'hyperparam_tuning_{base_model_name}',  # Separate directory for each model
            project_name=f'{base_model_name}_tuning'
        )

        # Load the data
        X_train, X_test, y_train, y_test = load_data()

        # Perform the hyperparameter search
        tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]

        # Append the best model to the list
        best_models.append(best_model)

    return best_models


def ensemble_predict(models, X_test):
    """
    Predict using an ensemble of models and combine the predictions.
    INPUT:
        models: List of fine-tuned models.
        X_test: Test data.
    OUTPUT:
        Final ensemble predictions.
    """
    # Get predictions from each model
    predictions = [model.predict(X_test) for model in models]

    # Averaging the predictions
    averaged_predictions = np.mean(predictions, axis=0)

    return averaged_predictions


def evaluate_ensemble(models, X_test, y_test):
    """
    Evaluate the ensemble on the test set.
    INPUT:
        models: List of fine-tuned models.
        X_test: Test data.
        y_test: True labels for the test data.
    OUTPUT:
        Prints the ensemble accuracy, precision, and recall.
    """
    # Get ensemble predictions
    y_pred_prob = ensemble_predict(models, X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Ensemble Accuracy: {accuracy}")
    print(f"Ensemble Precision: {precision}")
    print(f"Ensemble Recall: {recall}")


# Example usage
if __name__ == "__main__":
    # Load and fine-tune the ensemble of models
    models = load_pretrained_models()

    # Load test data
    X_train, X_test, y_train, y_test = load_data()

    # Evaluate the ensemble
    evaluate_ensemble(models, X_test, y_test)
