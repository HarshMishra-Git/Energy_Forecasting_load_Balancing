import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class AdvancedForecaster:
    def __init__(self, model_type='lstm', sequence_length=24):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_sequences(self, data):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
        
    def build_model(self, input_shape):
        """Build LSTM or GRU model"""
        model = Sequential()
        
        if self.model_type == 'lstm':
            model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(30, activation='relu'))
        else:  # GRU
            model.add(GRU(50, activation='relu', input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(30, activation='relu'))
            
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def train(self, df, epochs=50, batch_size=32):
        """Train the model and return predictions"""
        # Scale the data
        data = self.scaler.fit_transform(df[['energy_consumption']].values)
        
        # Create sequences
        X, y = self.create_sequences(data)
        
        # Split into train and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build and train model
        self.model = self.build_model((self.sequence_length, 1))
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Make predictions
        predictions = self.model.predict(X_val)
        predictions = self.scaler.inverse_transform(predictions)
        y_val_orig = self.scaler.inverse_transform(y_val.reshape(-1, 1))
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_val_orig, predictions),
            'rmse': np.sqrt(mean_squared_error(y_val_orig, predictions)),
            'mape': np.mean(np.abs((y_val_orig - predictions) / y_val_orig)) * 100
        }
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'history': history.history
        }
        
    def predict(self, data):
        """Generate predictions for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        scaled_data = self.scaler.transform(data)
        X, _ = self.create_sequences(scaled_data)
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)
