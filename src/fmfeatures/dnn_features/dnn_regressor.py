import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers.legacy import Adam

def get_change_estimate(data_path, target_column, 
                        model_builder, model_type):
    """
    Performs one-step-ahead forecasting of the target variable 
    using a TensorFlow neural network on financial time-series
    data with a fixed training window of 50 samples. 
    The DNN architecture is customizable via an external 
    model builder function.
    
    Parameters
    ----------
    data_path : str
        Path to the financial data CSV file.
    target_column : str
        Name of the column to be predicted.
    model_builder : callable
        A function that builds and returns a compiled TensorFlow Keras model. It should accept
        the input shape as its parameter.
    model_type: str
        The name of the model that will be used to obtain forecasts
        based on DNN (model_type must be 'dense', 'rnn', or 'lstm')
    
    Returns
    -------
    tuple of lists
        - `train_predictions`: List of predicted values for each training window.
        - `test_predictions`: List of one-step-ahead forecasted values.
    
    Examples
    --------
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
     
    # Define a custom model builder function
    def custom_model(input_shape):
         model = Sequential([
             Dense(64, activation='relu', input_shape=(input_shape,)),
             Dense(32, activation='relu'),
             Dense(1)  # Output layer for regression
         ])
         model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
         return model
     
    # Define the data with 100 samples to ensure enough data for training
    np.random.seed(42)
    data = {
         'Open': np.random.uniform(100, 200, 100).round(2),
         'High': np.random.uniform(100, 200, 100).round(2),
         'Low': np.random.uniform(100, 200, 100).round(2),
         'Close': np.random.uniform(100, 200, 100).round(2),
         'Volume': np.random.randint(1000, 5000, 100),
         'PriceChange': np.random.randint(-10, 11, 100)  # Target variable
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('financial_data.csv', index=False)
     
    # Use the function
    train_preds, test_preds = get_change_estimate('financial_data.csv', 'PriceChange', custom_model)
    print("Training Predictions:", train_preds)
    print("One-Step-Ahead Forecasts:", test_preds)
    """
    # Load financial data
    data = pd.read_csv(data_path)
    
    # Ensure the data is sorted by time if not already
    data = data.sort_index()
    
    # Check if there are enough samples
    window_size = 50
    if len(data) <= window_size:
        raise ValueError(f"Data must contain more than {window_size} samples for training.")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Normalize the features (excluding the target column)
    feature_columns = data.columns.drop(target_column)
    X_scaled = scaler.fit_transform(data[feature_columns].values)
    
    # Extract the target variable
    y = data[target_column].values
    
    train_predictions = []
    test_predictions = []
    
    # Iterate over the dataset to perform one-step-ahead forecasting
    for i in range(window_size, len(data)):
        # Define the training window
        X_train = X_scaled[i-window_size:i]
        y_train = y[i-window_size:i]
        
        # Define the test sample (current step)
        X_test = X_scaled[i].reshape(1, -1)
        y_true = y[i]
        
        # Build the TensorFlow neural network model using the external model_builder
        model = model_builder(X_train.shape[1], model_type=model_type)
        
        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        # Predict on the training window
        y_train_pred = model.predict(X_train).flatten()
        train_predictions.append(y_train_pred)
        
        # Predict one-step-ahead
        y_test_pred = model.predict(X_test)[0][0]
        test_predictions.append(y_test_pred)
    
    return train_predictions, test_predictions


def model_builder(input_shape, model_type='dense'):
    """
    Builds and compiles a TensorFlow Keras Sequential
    model with a specified architecture.
    
    Parameters
    ----------
    input_shape : int
        Number of input features.
    model_type : str, optional
        Type of model to build ('dense', 'rnn', 'lstm'), by default 'dense'
    
    Returns
    -------
    model : tensorflow.keras.Model
        Compiled Keras model ready for training.
    """
    model = Sequential()
    
    if model_type == 'dense':
        # Dense Neural Network
        model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
    elif model_type == 'rnn':
        # Simple Recurrent Neural Network
        model.add(SimpleRNN(64, activation='relu', input_shape=(input_shape, 1)))
        model.add(Dense(32, activation='relu'))
    elif model_type == 'lstm':
        # Long Short-Term Memory Network
        model.add(LSTM(64, activation='relu', input_shape=(input_shape, 1)))
        model.add(Dense(32, activation='relu'))
    else:
        raise ValueError("model_type must be 'dense', 'rnn', or 'lstm'")
    
    # Common output layer
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model