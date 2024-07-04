"""
    This file is handling the functions for the
    reflectance image. For CHIME and ENVI formats
"""
# Importing packages
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
import spectral.io.envi as envi
import os


#__________________________netCDF handle____________________________
def read_netcdf(path):
    """
    Reading the netcdf file
    :param path: path to the netcdf file
    :return: data cube of reflectance image, wavelength list
    """
    # Read netCDF image ! Cannot be accent in the path!
    ds_im = nc.Dataset(path)
    # Converting reflectance data into numpy array, scaling 1/10000
    # Scale is calculated from: image scale 1/100, difference between image
    # values and GPR RTM reflectance values
    np_refl = ds_im['l2a_BOA_rfl'][:]
    np_refl = np_refl.data
    data_refl = np_refl * 1 / 10000

    # Saving image wavelengths
    data_wavelength = ds_im['central_wavelength'][:]
    data_wavelength = data_wavelength.data

    return data_refl, data_wavelength


#__________________________________ENVI handle______________________________

def read_envi(path):
    """
    Read the ENVI format
    :param path: path of the ENVI file
    :return: data cube of reflectance image, wavelength list
    optionally returns latitude and longitude list if map information is available
    """
    # Open the ENVI file
    envi_image = envi.open(path, os.path.join(os.path.dirname(path),os.path.splitext(os.path.basename(path))[0]))

    # Load the data into a NumPy array
    data = envi_image.asarray()
    data = data * 1 / 10000

    # Storing all the metadata
    info = envi_image.metadata

    # Storing wavelengths
    data_wavelength = [int(float(wavelength)) for wavelength in envi_image.metadata['wavelength']]
    data_wavelength = np.array(data_wavelength)

    # Obtain lat,lon (transform UTM coordinates)
    if 'map info' in info:
        map_info = info['map info']
        lon = int(info['samples'])
        lat = int(info['lines'])
        longitude, latitude = get_lat_lon_envi(map_info, lon, lat)  # x,y
        return data, data_wavelength, longitude, latitude
    else:
        return data, data_wavelength


def get_lat_lon_envi(map_info, lon, lat):
    """
    Getting the latitude and longitude of the ENVI map
    :param map_info: contains map information
    :param lon: longitude of top right corner
    :param lat: latitude of top right corner
    :return: lists of latitude and longitude
    in degree coordinates
    """
    # Coordinates of the upper left corner
    xi = float(map_info[1])
    yi = float(map_info[2])
    xm = float(map_info[3])  # latitude
    ym = float(map_info[4])  # longitude
    dx = int(float(map_info[5]))
    dy = int(float(map_info[6]))
    # Adjust points to corner (1.5,1.5)
    if yi > 1.5:
        ym += (yi * dy) - dy
    if xi > 1.5:
        xm -= (xi * dy) - dx
    max_latlon = max(lat, lon)
    x_vector = xm + np.arange(max_latlon) * dx
    y_vector = np.flip(ym - np.arange(max_latlon) * dy)

    # Define the projection parameters
    utm_zone = int(map_info[7])
    utm_hemisphere = map_info[8]  # Assuming the hemisphere is North
    datum = map_info[9].replace("-", "")
    utm_proj_string = f'+proj=utm +zone={utm_zone} +{utm_hemisphere} +datum={datum}'

    # Create a pyproj projection object
    utm_proj = Proj(utm_proj_string)

    # Convert the UTM coordinates to latitude and longitude
    longitude, latitude = utm_proj(x_vector, y_vector, inverse=True)

    # Cut values if needed
    if lat < lon:
        latitude = latitude[:lat]
    elif lon < lat:
        longitude = longitude[:lon]

    #TODO: for testing
    #print("Latitude:", latitude, len(latitude))
    #print("Longitude:", longitude, len(longitude))

    return longitude, latitude  # x,y


#_____________________________________ Plotting images ________________________________________

def show_reflectance_img(data_refl, data_wavelength):
    """
    Showing the image read
    :param data_refl: data cube of reflectance (y,x,dim)
    :param data_wavelength: list of wavelengths
    :return: no return value just plotting the image
    """
    # Defining wavelength RGB
    indexes = np.zeros(3)
    values_to_find = [639, 547, 463]
    # Find the index closest to each value
    for i, value in enumerate(values_to_find):
        closest_index = np.abs(data_wavelength - value).argmin()
        indexes[i] = closest_index
    idx_int = indexes.astype(np.uint8)
    data_r_for_show = data_refl[:, :, idx_int]
    # Normalise image
    data_r_for_show_norm = (data_r_for_show - np.min(data_r_for_show)) / (
            np.max(data_r_for_show) - np.min(data_r_for_show))
    # Showing the image
    plt.imshow(data_r_for_show_norm, interpolation='nearest')
    plt.title('Reflectance image (RGB)')
    plt.colorbar()
    plt.show()



#________________________________ Exporting results ___________________________________