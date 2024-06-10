"""
This code is applied to the NOAA data 
To check the decadal variation of mean SST


"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

ds = xr.open_dataset('Data_noaa_copernicus/noaa_avhrr/noaasst1982_2023_data.nc')

decades = [
    ('1982-01-01', '1991-12-31'),
    ('1992-01-01', '2001-12-31'),
    ('2002-01-01', '2011-12-31'),
    ('2012-01-01', '2023-12-31')
]

# Compute the mean SST for each decade
decadal_means = []
decade_labels = []
all_sst_values = []  
for start, end in decades:
    decadal_mean = ds.sel(time=slice(start, end)).mean(dim='time')
    decadal_means.append(decadal_mean)
    decade_labels.append(f'{start[:4]}-{end[:4]}')
    all_sst_values.extend(decadal_mean.sst.values.flatten())

mean_sst_values = [decadal_mean.sst.mean().item() for decadal_mean in decadal_means]

decadal_changes = np.diff(mean_sst_values, prepend=mean_sst_values[0])

plt.figure(figsize=(12, 6))
bars = plt.bar(decade_labels, mean_sst_values, color='goldenrod', edgecolor='black')
plt.title('Decadal Sea Surface Temperature Variation (1982-2023)', fontsize=16, fontweight='bold')
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Mean SST (°C)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

for bar, change in zip(bars, decadal_changes):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{change:+.2f}°C', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

sst_min = np.nanmin(all_sst_values)
sst_max = np.nanmax(all_sst_values)

for i, (start, end) in enumerate(decades):
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=1)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.set_extent([ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()], crs=ccrs.PlateCarree())
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    sst_values = decadal_means[i].sst.values
    sst_values = np.ma.masked_invalid(sst_values)
    cmap = plt.get_cmap('jet')
    plt.pcolormesh(ds.lon, ds.lat, sst_values, cmap=cmap, transform=ccrs.PlateCarree(),
                   norm=colors.Normalize(vmin=sst_min, vmax=sst_max))
    cbar = plt.colorbar(ax=ax, orientation='vertical', pad=0.02, aspect=50)
    cbar.set_label('SST (°C)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.title(f'Sea Surface Temperature ({start[:4]}-{end[:4]})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
