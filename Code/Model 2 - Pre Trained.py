import pandas as pd
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import precision_score, accuracy_score
import numpy as np

# Image size and number of classes
img_rows, img_cols = 256, 256  
channels = 3  
classes = 5 

# Load the preprocessed data saved as .npy files
def load_data():
    """
    Load the preprocessed data from NumPy files.
    """
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    
    # Normalize image data to 0-1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
    return X_train, X_test, y_train, y_test

# Define model architectures
def build_alexnet(hp):
    """Define AlexNet architecture with hyperparameter tuning."""
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(img_rows, img_cols, channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(Flatten())
    
    units = hp.Int('units', min_value=512, max_value=2048, step=256)
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(classes, activation='softmax'))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_vggnet(hp, depth):
    """Define VGGNet (VGG-16, VGG-19, or VGG-s) architecture with hyperparameter tuning."""
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    if depth >= 16:  # Add extra layers for VGG-16, VGG-19
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
    if depth == 19:  # Add extra layers for VGG-19
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
    model.add(Flatten())
    
    units = hp.Int('units', min_value=512, max_value=2048, step=256)
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(classes, activation='softmax'))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_googlenet(hp):
    """Define GoogleNet (InceptionV1) architecture with hyperparameter tuning."""
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    units = hp.Int('units', min_value=512, max_value=2048, step=256)
    x = Dense(units, activation='relu')(x)
    x = Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
    
    predictions = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_resnet(hp):
    """Define ResNet architecture with hyperparameter tuning."""
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    units = hp.Int('units', min_value=512, max_value=2048, step=256)
    x = Dense(units, activation='relu')(x)
    x = Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
    
    predictions = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hyperparameter tuning function
def tune_model(build_fn, model_name, X_train, y_train, X_test, y_test):
    """Tune the model using Keras Tuner."""
    tuner = kt.RandomSearch(
        build_fn,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=2,
        directory=f'hyperparam_tuning_{model_name}',
        project_name=f'{model_name}_tuning'
    )
    
    tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Predict and calculate precision and accuracy
    y_pred_prob = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    precision = precision_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    return precision, accuracy

# Main function to train and evaluate all models
def train_and_evaluate_models():
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=['Model', 'Precision', 'Accuracy'])
    
    # Define a list of model configurations
    models = [
        ('AlexNet', build_alexnet),
        ('VGGNet-s', lambda hp: build_vggnet(hp, depth=11)),  # VGGNet-s is a simplified version of VGG
        ('VGGNet-16', lambda hp: build_vggnet(hp, depth=16)),
        ('VGGNet-19', lambda hp: build_vggnet(hp, depth=19)),
        ('GoogleNet', build_googlenet),
        ('ResNet', build_resnet)
    ]
    
    # Train and tune each model
    for model_name, build_fn in models:
        print(f"Training and tuning {model_name}...")
        precision, accuracy = tune_model(build_fn, model_name, X_train, y_train, X_test, y_test)
        
        # Add results to the DataFrame
        results_df = results_df.append({'Model': model_name, 'Precision': precision, 'Accuracy': accuracy}, ignore_index=True)
    
    # Save the results to a CSV file
    #results_df.to_csv('model_results.csv', index=False)
    print("Model training and evaluation completed.")
    print(results_df)

if __name__ == "__main__":
    train_and_evaluate_models()
