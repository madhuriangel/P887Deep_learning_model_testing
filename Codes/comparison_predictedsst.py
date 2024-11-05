import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Load datasets
predicted_sst = xr.open_dataset('noaa_newfile/mi_roms_data.nc')
dl_predicted_sst = xr.open_dataset('Data_noaa_copernicus/lstmpred2022_best_scale.nc')
actual_sst = xr.open_dataset('Data_noaa_copernicus/noaa_avhrr/noaa_icesmi_combinefile2022.nc')

# Extract SST data
pred_sst = predicted_sst['sst']
dl_predsst = dl_predicted_sst['sst']
act_sst = actual_sst['sst']

pred_sst_flat = pred_sst.values.flatten()
dl_predsst_flat = dl_predsst.values.flatten()
act_sst_flat = act_sst.values.flatten()

# Mask the NaN values
mask = ~np.isnan(pred_sst_flat) & ~np.isnan(act_sst_flat) & ~np.isnan(dl_predsst_flat)

pred_sst_filtered = pred_sst_flat[mask]
dl_predsst_filtered = dl_predsst_flat[mask]
act_sst_filtered = act_sst_flat[mask]

# Calculate statistical metrics after filtering NaNs for MI Model
mae_mimodel = mean_absolute_error(act_sst_filtered, pred_sst_filtered)
rmse_mimodel = np.sqrt(mean_squared_error(act_sst_filtered, pred_sst_filtered))
bias_mimodel = np.mean(pred_sst_filtered - act_sst_filtered)
correlation_mimodel, _ = pearsonr(pred_sst_filtered, act_sst_filtered)

print(f"MI Model - Mean Absolute Error (MAE): {mae_mimodel:.3f}")
print(f"MI Model - Root Mean Square Error (RMSE): {rmse_mimodel:.3f}")
print(f"MI Model - Bias: {bias_mimodel:.3f}")
print(f"MI Model - Correlation Coefficient: {correlation_mimodel:.3f}")

# Calculate statistical metrics for DL Model
mae_dl = mean_absolute_error(act_sst_filtered, dl_predsst_filtered)
rmse_dl = np.sqrt(mean_squared_error(act_sst_filtered, dl_predsst_filtered))
bias_dl = np.mean(dl_predsst_filtered - act_sst_filtered)
correlation_dl, _ = pearsonr(dl_predsst_filtered, act_sst_filtered)

print(f"DL Model - Mean Absolute Error (MAE): {mae_dl:.3f}")
print(f"DL Model - Root Mean Square Error (RMSE): {rmse_dl:.3f}")
print(f"DL Model - Bias: {bias_dl:.3f}")
print(f"DL Model - Correlation Coefficient: {correlation_dl:.3f}")

x_axis = np.arange(len(act_sst_filtered))

# Time Series plot for the nearest location to lat=50, lon=350
plt.figure(figsize=(10,6))
plt.plot(pred_sst.sel(lat=50, lon=350, method='nearest'), label='MI Model Predicted SST', color='#042759')  # Oxford Blue
plt.plot(dl_predsst.sel(lat=50, lon=350, method='nearest'), label='DL Model Predicted SST', color='#7ca982')  # Cambridge Blue
plt.plot(act_sst.sel(lat=50, lon=350, method='nearest'), label='Actual SST', color='#DAA520')  # Goldenrod

plt.xlabel('Time')
plt.ylabel('Sea Surface Temperature (°C)')
plt.title('Time Series of Predicted vs Actual SST at Nearest Grid Point to Lat=50, Lon=350')
plt.legend()
plt.grid()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.ravel()

# DL Model Predicted SST
axes[0].set_title('DL Model Predicted SST (Day 1)')
dl_plot = axes[0].pcolormesh(dl_predsst.lon, dl_predsst.lat, dl_predsst.isel(time=0), 
                             cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(dl_plot, ax=axes[0], orientation='vertical', label='SST (°C)')
axes[0].add_feature(cfeature.COASTLINE, zorder=10)
axes[0].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
axes[0].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# Actual SST
axes[1].set_title('Actual SST (Day 1)')
actual_plot = axes[1].pcolormesh(act_sst.lon, act_sst.lat, act_sst.isel(time=0), 
                                 cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(actual_plot, ax=axes[1], orientation='vertical', label='SST (°C)')
axes[1].add_feature(cfeature.COASTLINE, zorder=10)
axes[1].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
axes[1].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# SST Difference (DL Predicted - Actual)
axes[2].set_title('SST Difference (DL Predicted - Actual) on Day 1')
diff_plot = axes[2].pcolormesh(dl_predsst.lon, dl_predsst.lat, dl_predsst.isel(time=0) - act_sst.isel(time=0),
                               cmap='bwr', vmin=-2, vmax=2, transform=ccrs.PlateCarree())
plt.colorbar(diff_plot, ax=axes[2], orientation='vertical', label='SST Difference (°C)')
axes[2].add_feature(cfeature.COASTLINE, zorder=10)
axes[2].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
axes[2].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

plt.tight_layout()
plt.show()

# Scatter plot of DL Predicted vs Actual SST
plt.figure(figsize=(8,8))
plt.scatter(dl_predsst_filtered, act_sst_filtered, alpha=0.5, color='#7ca982')  # Cambridge Blue
plt.xlabel('DL Predicted SST (°C)')
plt.ylabel('Actual SST (°C)')
plt.title('Scatter Plot: DL Predicted vs Actual SST')
plt.plot([act_sst_filtered.min(), act_sst_filtered.max()], [act_sst_filtered.min(), act_sst_filtered.max()], 'r--')
plt.grid()
plt.show()

# Spatially-averaged metrics for DL Model across all time steps
mae_space = np.mean(np.abs(dl_predsst - act_sst), axis=0)
rmse_space = np.sqrt(np.mean((dl_predsst - act_sst)**2, axis=0))
bias_space = np.mean(dl_predsst - act_sst, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# MAE
axes[0].set_title('DL Model MAE Across All Time Steps')
mae_plot = axes[0].pcolormesh(dl_predsst.lon, dl_predsst.lat, mae_space, cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(mae_plot, ax=axes[0], orientation='vertical', label='MAE (°C)')
axes[0].add_feature(cfeature.COASTLINE, zorder=10)
axes[0].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
axes[0].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# RMSE
axes[1].set_title('DL Model RMSE Across All Time Steps')
rmse_plot = axes[1].pcolormesh(dl_predsst.lon, dl_predsst.lat, rmse_space, cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(rmse_plot, ax=axes[1], orientation='vertical', label='RMSE (°C)')
axes[1].add_feature(cfeature.COASTLINE, zorder=10)
axes[1].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
axes[1].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# Bias
axes[2].set_title('DL Model Bias Across All Time Steps')
bias_plot = axes[2].pcolormesh(dl_predsst.lon, dl_predsst.lat, bias_space, cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(bias_plot, ax=axes[2], orientation='vertical', label='Bias (°C)')
axes[2].add_feature(cfeature.COASTLINE, zorder=10)
axes[2].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
axes[2].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

plt.tight_layout()
plt.show()

# Generate a land-sea mask using the actual SST dataset
sea_mask = ~np.isnan(act_sst.isel(time=0))  # True for sea, False for land

mean_pred_sst = pred_sst.where(sea_mask).mean(dim='time', skipna=True)
mean_dl_predsst = dl_predsst.where(sea_mask).mean(dim='time', skipna=True)
mean_act_sst = act_sst.where(sea_mask).mean(dim='time', skipna=True)

# Plot yearly mean SST for MI Model, DL Model, and Actual SST
fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# MI Model
ax[0].set_title('MI Model Yearly Mean SST (2022)')
mi_plot = ax[0].pcolormesh(pred_sst.lon, pred_sst.lat, mean_pred_sst, cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(mi_plot, ax=ax[0], orientation='vertical', label='SST (°C)')
ax[0].add_feature(cfeature.COASTLINE, zorder=10)
ax[0].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
ax[0].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# DL Model
ax[1].set_title('DL Model Yearly Mean SST (2022)')
dl_plot = ax[1].pcolormesh(dl_predsst.lon, dl_predsst.lat, mean_dl_predsst, cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(dl_plot, ax=ax[1], orientation='vertical', label='SST (°C)')
ax[1].add_feature(cfeature.COASTLINE, zorder=10)
ax[1].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
ax[1].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# Actual SST
ax[2].set_title('Actual Yearly Mean SST (2022)')
actual_plot = ax[2].pcolormesh(act_sst.lon, act_sst.lat, mean_act_sst, cmap='coolwarm', transform=ccrs.PlateCarree())
plt.colorbar(actual_plot, ax=ax[2], orientation='vertical', label='SST (°C)')
ax[2].add_feature(cfeature.COASTLINE, zorder=10)
ax[2].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
ax[2].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

plt.tight_layout()
plt.show()

# Calculate differences between actual and model SST yearly mean
diff_mi_act = mean_act_sst - mean_pred_sst
diff_dl_act = mean_act_sst - mean_dl_predsst

# Plot differences for MI Model and DL Model
fig, ax = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# MI Model - Actual Difference
ax[0].set_title('Difference: Actual - MI Model (2022)')
mi_diff_plot = ax[0].pcolormesh(pred_sst.lon, pred_sst.lat, diff_mi_act, cmap='bwr', vmin=-2, vmax=2, transform=ccrs.PlateCarree())
plt.colorbar(mi_diff_plot, ax=ax[0], orientation='vertical', label='SST Difference (°C)')
ax[0].add_feature(cfeature.COASTLINE, zorder=10)
ax[0].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
ax[0].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

# DL Model - Actual Difference
ax[1].set_title('Difference: Actual - DL Model (2022)')
dl_diff_plot = ax[1].pcolormesh(dl_predsst.lon, dl_predsst.lat, diff_dl_act, cmap='bwr', vmin=-2, vmax=2, transform=ccrs.PlateCarree())
plt.colorbar(dl_diff_plot, ax=ax[1], orientation='vertical', label='SST Difference (°C)')
ax[1].add_feature(cfeature.COASTLINE, zorder=10)
ax[1].add_feature(cfeature.BORDERS, linestyle=':', zorder=10)
ax[1].add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)

plt.tight_layout()
plt.show()
