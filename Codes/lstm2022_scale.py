import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

# Load full data
data = xr.open_dataset('noaa_icesmi_combinefile_FINAL.nc')
sst = data['sst'].values

# Define training and testing time periods
train_start = np.datetime64('1982-01-01')
train_end = np.datetime64('2019-12-31')  # Training data up to end of 2019
test_start = np.datetime64('2020-01-01')  # Test data starts from 2020
test_end = np.datetime64('2023-12-31')    # Test data up to end of 2023

# Extract training and test datasets based on time selection
train_data = sst[(data['time'] >= train_start) & (data['time'] <= train_end)]
test_data = sst[(data['time'] >= test_start) & (data['time'] <= test_end)]

# Flatten spatial dimensions, reshape to 2D arrays, and normalize
scaler = StandardScaler()
train_data_reshaped = train_data.reshape(train_data.shape[0], -1)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

# Handle NaNs by creating masks for valid (non-NaN) grid points
mask_tr = ~np.isnan(train_data).any(axis=0)  # 2D mask (lat, lon)
mask_te = ~np.isnan(test_data).any(axis=0)  # 2D mask (lat, lon)

# Flatten spatial dimensions and normalize only non-NaN grid points
train_data_flat = train_data_reshaped[:, mask_tr.ravel()]
test_data_flat = test_data_reshaped[:, mask_te.ravel()]

# Fit scaler on training data and transform both training and test data
train_data_norm = scaler.fit_transform(train_data_flat)
test_data_norm = scaler.transform(test_data_flat)

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Hyperparameters
learning_rates = [0.0001]
batch_sizes = [32, 64]
lstm_units = [(512, 256, 128),(1024, 512, 256)]
dropout_rates = [0,0.1]
epochs_list = [300]
sequence_lengths = [180, 365]

best_rmse = float('inf')
best_config = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_units[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_units[0], hidden_units[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_units[1], hidden_units[2], batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_units[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

train_loss_history = []
val_loss_history = []

# Training and hyperparameter tuning loop
for lr in learning_rates:
    for bs in batch_sizes:
        for units in lstm_units:
            for dr in dropout_rates:
                for epochs in epochs_list:
                    for seq_len in sequence_lengths:
                        train_sequences = create_sequences(train_data_norm, seq_len)
                        test_sequences = create_sequences(test_data_norm, seq_len)
                        x_train, x_val = train_test_split(train_sequences, test_size=0.2, random_state=42)

                        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
                        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32))
                        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

                        model = LSTMModel(input_size=x_train.shape[2], hidden_units=units, dropout_rate=dr).to(device)
                        criterion = nn.MSELoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        patience = 10
                        patience_counter = 0
                        best_val_loss = float('inf')

                        for epoch in range(epochs):
                            model.train()
                            train_losses = []
                            for batch in train_loader:
                                optimizer.zero_grad()
                                inputs = batch[0].to(device)
                                outputs = model(inputs)
                                loss = criterion(outputs, inputs)
                                loss.backward()
                                optimizer.step()
                                train_losses.append(loss.item())

                            model.eval()
                            val_losses = []
                            with torch.no_grad():
                                for batch in val_loader:
                                    inputs = batch[0].to(device)
                                    outputs = model(inputs)
                                    loss = criterion(outputs, inputs)
                                    val_losses.append(loss.item())

                            val_loss = np.mean(val_losses)
                            train_loss = np.mean(train_losses)
                            train_loss_history.append(train_loss)
                            val_loss_history.append(val_loss)
                            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                torch.save(model.state_dict(), 'model_save/best_model.pth')
                                patience_counter = 0
                            else:
                                patience_counter += 1

                            if patience_counter >= patience:
                                print("Early stopping")
                                break

                        # Load best model for evaluation
                        model.load_state_dict(torch.load('model_save/best_model.pth'))

                        # Predict on test data
                        model.eval()
                        test_dataset = TensorDataset(torch.tensor(test_sequences, dtype=torch.float32))
                        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
                        predicted_data = []

                        with torch.no_grad():
                            for batch in test_loader:
                                inputs = batch[0].to(device)
                                outputs = model(inputs)
                                predicted_data.append(outputs.cpu().numpy())

                        # Only concatenate if `predicted_data` contains data
                        if predicted_data:
                            predicted_data = np.concatenate(predicted_data, axis=0)
                        else:
                            print("Warning: No data in predicted_data to concatenate. Skipping further steps.")
                            continue  # Skip further processing if predicted_data is empty

                        # Inverse transform to the original scale
                        test_data_flat = test_sequences.reshape(-1, test_sequences.shape[-1])
                        predicted_data_flat = predicted_data.reshape(-1, predicted_data.shape[-1])
                        test_data_orig_flat = scaler.inverse_transform(test_data_flat)
                        predicted_data_orig_flat = scaler.inverse_transform(predicted_data_flat)

                        # Calculate indices for 2022 within the test period
                        date_2022_start = np.datetime64('2022-01-01')
                        date_2022_end = np.datetime64('2022-12-31')
                        index_2022_start = (date_2022_start - test_start).astype(int)
                        index_2022_end = (date_2022_end - test_start).astype(int)

                        # Use these indices for slicing the 2022 period
                        start_pred_idx = index_2022_start - seq_len
                        end_pred_idx = index_2022_end  # Ending index for predictions covering 2022


                        # Ensure indices are within bounds
                        if start_pred_idx < 0 or end_pred_idx > predicted_data_flat.shape[0]:
                            print("Sequence length too long to get predictions for 2022. Skipping.")
                            continue  # Skip this configuration if indices are invalid

                        # Reconstruct data for valid lat-lon grid points
                        lat_idx, lon_idx = np.where(mask_te)
                        test_data_full_2022 = np.full((365, 28, 36), np.nan)
                        predicted_data_full_2022 = np.full((365, 28, 36), np.nan)

                        for t in range(365):
                            test_data_full_2022[t, lat_idx, lon_idx] = test_data_orig_flat[start_pred_idx+t, :]
                            predicted_data_full_2022[t, lat_idx, lon_idx] = predicted_data_orig_flat[start_pred_idx+t, :]

                        # Calculate RMSE for 2022
                        mse = np.nanmean((test_data_full_2022 - predicted_data_full_2022) ** 2)
                        rmse = np.sqrt(mse)
                        print(f"Config: lr={lr}, bs={bs}, units={units}, dr={dr}, epochs={epochs}, seq_len={seq_len}, RMSE={rmse}")

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_config = {
                                'learning_rate': lr,
                                'batch_size': bs,
                                'lstm_units': units,
                                'dropout_rate': dr,
                                'epochs': epochs,
                                'sequence_length': seq_len,
                                'rmse': rmse
                            }
                            best_predicted_data = predicted_data_full_2022

# Save best predicted data as a NetCDF file
print(f"Best Configuration: {best_config}")
if best_predicted_data is not None:
    predicted_ds = xr.Dataset(
        {
            'sst': (('time', 'lat', 'lon'), best_predicted_data)
        },
        coords={
            'time': pd.date_range(start='2022-01-01', periods=365),
            'lat': data['lat'].values,
            'lon': data['lon'].values
        }
    )
    predicted_ds.to_netcdf('lstmpred2022_best_scale.nc')
    print("Predicted data from the best model saved as 'lstmpred2022_best_scale.nc'")

# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig('gridded_images/training_vs_validation_scale.png')
plt.close()

# RMSE spatial distribution plot for 2022
latitude = data['lat'].values
longitude = data['lon'].values
rmse_grid = np.sqrt(np.nanmean((test_data_full_2022 - predicted_data_full_2022) ** 2, axis=0))


plt.figure(figsize=(10, 6))
plt.imshow(rmse_grid, cmap='jet', origin='lower',
           extent=[longitude.min(), longitude.max(), latitude.min(), latitude.max()])
plt.title('RMSE Spatial Distribution for 2022')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='RMSE')
plt.savefig('gridded_images/RMSE_spatial_scale_2022.png')
plt.close()

# MAPE and accuracy
mask = test_data_full_2022 != 0
mape = np.nanmean(np.abs((test_data_full_2022[mask] - predicted_data_full_2022[mask]) / test_data_full_2022[mask])) * 100
accuracy = 100 - mape
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Prediction Accuracy: {accuracy}%")


