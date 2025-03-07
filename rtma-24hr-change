import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from datetime import datetime
import datetime as dt
from metpy.units import units
import matplotlib.colors as col
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import metpy.calc as mpcalc
import supplementary_tools as spt
from metpy.plots import USCOUNTIES
from gradient import gradient

# Helper function to find the nearest index in an array
def find_nearest_index(array, val):
    return np.abs(array - val).argmin()

# Define the output directory for the generated plot
outputPath = '../PythonFigures/'

# Get the date and time of the initialization for the RTMA data
mdate1 = spt.get_init_time('RTMA')[0]
init_hour1 = '21'

# URLs to fetch the RTMA data for the given date and time
url1 = f'http://nomads.ncep.noaa.gov:80/dods/rtma2p5/rtma2p5{mdate1}/rtma2p5_anl_{init_hour1}z'
print(url1)

# Calculate the date for the previous day and construct the URL for the second dataset
mdate2 = datetime.strptime(mdate1, '%Y%m%d') - dt.timedelta(hours=24)
mdate2 = datetime.strftime(mdate2, '%Y%m%d')
url2 = f'http://nomads.ncep.noaa.gov:80/dods/rtma2p5/rtma2p5{mdate2}/rtma2p5_anl_{init_hour1}z'
print(url2)

# Open the datasets from the URLs using xarray
ds1 = xr.open_dataset(url1)
ds2 = xr.open_dataset(url2)

# Extract time and coordinate information from the datasets
times1 = ds1['tmp2m'].metpy.time
times2 = ds2['tmp2m'].metpy.time
init_time = ds1['time'][0]

# Define the latitude and longitude grid for the map
lats = np.arange(25, 55, 0.25)
lons = np.arange(260, 310, 0.25)

# Parse the data using MetPy
data1 = ds1.metpy.parse_cf()
data2 = ds2.metpy.parse_cf()

# Rename the data variables to more descriptive names
data1 = data1.rename({'tmp2m': 'temperature1'})
data2 = data2.rename({'tmp2m': 'temperature2'})

# Extract temperature data and convert from Kelvin to Fahrenheit
t2m1 = data1['temperature1'].squeeze()
t2m1 = ((t2m1 - 273.15) * (9./5.)) + 32.
t2m2 = data2['temperature2'].squeeze()
t2m2 = ((t2m2 - 273.15) * (9./5.)) + 32.

# Calculate the temperature change between the two datasets (24-hour change)
dTdt = t2m1 - t2m2

# Read temperature color values from a file and set the color levels
colors = []
with open('tchange.txt', 'r') as f:
    for line in f:
        colors.append(line.strip())

clevs = np.arange(-70, 71, 1)

# Create a custom colormap for temperature changes
temp_colors = gradient(
    [[colors[0], -70.0], [colors[1], -50.0]],
    [[colors[1], -50.0], [colors[2], -25.0]],
    [[colors[2], -25.0], [colors[3], -10.0]],
    [[colors[3], -10.0], [colors[4], 0.0]],
    [[colors[4], 0.0], [colors[5], 10.0]],
    [[colors[5], 10.0], [colors[6], 25.0]],
    [[colors[6], 25.0], [colors[7], 50.0]],
    [[colors[7], 50.0], [colors[8], 70.0]]
)
cmap = temp_colors.get_cmap(clevs)
norm = col.BoundaryNorm(clevs, cmap.N)

# Set up map features (land, sea, and lakes)
land_mask = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='#fbf5ea')
sea_mask = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', edgecolor='face', facecolor='#edfbff')
lake_mask = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='face', facecolor='#edfbff')

# Define map extents for different regions
sub_w_ne, sub_e_ne, sub_n_ne, sub_s_ne = 275, 293.5, 47.5, 37

# Create a figure for plotting
fig = plt.figure(figsize=(18, 12), dpi=125)
gs = GridSpec(12, 36, figure=fig)

# Set up the main plot axis with the map projection
ax = plt.subplot(gs[:-1, :], projection=t2m1.metpy.cartopy_crs)
ax.add_feature(sea_mask, zorder=10)
ax.add_feature(land_mask, zorder=0)
ax.add_feature(lake_mask, zorder=10)
ax.coastlines(resolution='10m', linewidth=2.5, color='black', zorder=16)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=2, color='black', zorder=15)
ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=2, edgecolor='black', zorder=15)
ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=1, edgecolor='black', zorder=13)

# Set the title and add the time information
dtfs = str(times1.dt.strftime('%Y-%m-%d_%H%MZ').item())
print(f"Date: {dtfs}")
fig.suptitle(r"RTMA | 24-hr 2m Temperature Change (°F)", transform=ax.transAxes, fontweight="bold", fontsize=24, y=1.084, x=0, ha="left", va="top")
plt.title(f"Init: {init_time.dt.strftime('%Y-%m-%d %H:%MZ').item()} | Valid: {times1.dt.strftime('%Y-%m-%d %H:%MZ').item()}", x=0.0, y=1.015, fontsize=20, loc="left", transform=ax.transAxes)

# Plot the temperature change data on the map
temp = ax.contourf(data1['temperature1'].metpy.coordinates('x'), data1['temperature1'].metpy.coordinates('y'), dTdt, clevs, cmap=cmap, norm=norm, extend='both', transform=t2m1.metpy.cartopy_crs)

# Add colorbar for temperature change
cb1 = plt.colorbar(temp, orientation='horizontal', ticks=np.arange(-30, 31, 10), shrink=0.79, aspect=70, pad=0.02)
cb1.set_label('24-hr 2m Temperature Change (°F)', fontsize=16)
cb1.ax.tick_params(labelsize=14)

# Add the author text
fig.text(0.742, 0.25, 'Plot by Matthew Eovino', fontsize=18)

# Plot the cities and temperature data at specified locations
lats_i, lons_i, cities = [], [], []
with open('lats.txt', 'r') as f:
    lats_i = [line.strip() for line in f]
with open('lons.txt', 'r') as f:
    lons_i = [line.strip() for line in f]
with open('cities.txt', 'r') as f:
    cities = [line.strip() for line in f]

for i in range(len(lats_i)):
    lat_i = find_nearest_index(data1['temperature1'].metpy.coordinates('y'), float(lats_i[i]))
    lon_i = find_nearest_index(data1['temperature1'].metpy.coordinates('x'), float(lons_i[i]))
    station = round(float(dTdt[lat_i, lon_i]))
    if station > 0:
        station = "+" + str(station)
    plt.text(float(lons_i[i]), float(lats_i[i]) + 0.25, str(station), weight='heavy', fontsize=22, color='#F4DB81', verticalalignment='center', clip_on=True, horizontalalignment='center', transform=t2m1.metpy.cartopy_crs, zorder=30).set_path_effects([path_effects.Stroke(linewidth=2.2, foreground='black'), path_effects.Normal(), path_effects.SimpleLineShadow()])
    plt.text(float(lons_i[i]), float(lats_i[i]), str(cities[i]), weight='heavy', fontsize=12, color='white', verticalalignment='center', clip_on=True, horizontalalignment='center', transform=t2m1.metpy.cartopy_crs, zorder=30).set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal(), path_effects.SimpleLineShadow()])

# Set map extent and save the plot
ax.set_extent((sub_w_ne, sub_e_ne, sub_s_ne, sub_n_ne))
plt.tight_layout()
plt.savefig(f'{outputPath}RTMA_24tempchange_{init_hour1}z.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

print("---> Finished!")
