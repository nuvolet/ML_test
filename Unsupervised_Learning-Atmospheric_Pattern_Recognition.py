'''
Unsupervised Learning: Atmospheric Pattern Recognition

Problem:
Identifying recurring weather patterns and atmospheric regimes without labeled data.

Real-World Application:
The European Centre for Medium-Range Weather Forecasts (ECMWF) uses similar clustering techniques to identify recurring circulation patterns over Europe. 
These weather regimes help meteorologists understand atmospheric blocking events, heatwaves, and cold spells. 
The insights gained improve medium-range forecasts and seasonal outlooks.
'''

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from eofs.xarray import Eof

# Load geopotential height data (e.g., 500 hPa level)
ds = xr.open_dataset('/path/to/geopotential_height_data.nc')
z500 = ds['z500']  # Geopotential height at 500 hPa

# Select North Atlantic region
z500_region = z500.sel(latitude=slice(20, 80), longitude=slice(-90, 40))

# Remove seasonal cycle to focus on weather patterns
climatology = z500_region.groupby('time.dayofyear').mean('time')
z500_anomaly = z500_region.groupby('time.dayofyear') - climatology

# Prepare data for EOF/PCA
# Reshape to [time, space]
nlat, nlon = len(z500_anomaly.latitude), len(z500_anomaly.longitude)
z500_flat = z500_anomaly.stack(z=('latitude', 'longitude'))

# Apply EOF/PCA analysis
solver = Eof(z500_anomaly)
eofs = solver.eofs(neofs=10)  # Get first 10 EOFs
pcs = solver.pcs(npcs=10)     # Get first 10 principal components

# Standardize the PC time series
scaler = StandardScaler()
pcs_standardized = scaler.fit_transform(pcs.values)

# Apply K-means clustering to identify weather regimes
n_clusters = 4  # Typical number of North Atlantic regimes
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(pcs_standardized)

# Add cluster labels to the original dataset
z500_anomaly['cluster'] = ('time', labels)

# Calculate composite maps for each regime
regime_composites = [z500_anomaly.where(z500_anomaly.cluster == i).mean('time') for i in range(n_clusters)]

# Plot the weather regimes
fig = plt.figure(figsize=(15, 12))
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)

for i, composite in enumerate(regime_composites):
    ax = fig.add_subplot(2, 2, i+1, projection=proj)
    
    # Plot regime pattern
    im = composite.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                cmap='RdBu_r', levels=15, add_colorbar=False)
    
    # Calculate frequency
    regime_freq = (labels == i).sum() / len(labels) * 100
    
    # Add coastlines and title
    ax.coastlines()
    ax.set_title(f'Regime {i+1}: {regime_freq:.1f}% frequency')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Geopotential Height Anomaly (m)')

plt.tight_layout()
plt.savefig('weather_regimes.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze transitions between regimes
transitions = np.zeros((n_clusters, n_clusters))
for i in range(len(labels) - 1):
    transitions[labels[i], labels[i+1]] += 1

# Normalize to get transition probabilities
transition_prob = transitions / transitions.sum(axis=1, keepdims=True)

print("Transition Probability Matrix:")
print(pd.DataFrame(transition_prob, 
                  index=[f'From Regime {i+1}' for i in range(n_clusters)],
                  columns=[f'To Regime {i+1}' for i in range(n_clusters)]))

# Analyze persistence of regimes
persistence = np.zeros(n_clusters)
for i in range(n_clusters):
    runs = [len(list(g)) for k, g in itertools.groupby(labels) if k == i]
    persistence[i] = np.mean(runs) if runs else 0

print("\nAverage Persistence (days):")
for i in range(n_clusters):
    print(f"Regime {i+1}: {persistence[i]:.1f} days")