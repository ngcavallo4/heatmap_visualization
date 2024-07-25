import numpy as np

def gps_coords_to_meters(lons, lats): 

    ref_lat = np.min(lats)
    ref_lon = np.min(lons)

    long_m_array = []
    lat_m_array = []
    
    for lat, lon in zip(lats, lons):
        long_m, lat_m = haversine(lat, lon, ref_lat, ref_lon)
        long_m_array.append(long_m)
        lat_m_array.append(lat_m)

    return np.array(long_m_array), np.array(lat_m_array)

def haversine(lat, lon, ref_lat, ref_lon):

    # Earth radius in meters
        R = 6378137.0

        # Convert degrees to radians
        lat_v = np.deg2rad(lat)
        lon_v = np.deg2rad(lon)
        lat_u = np.deg2rad(ref_lat)
        lon_u = np.deg2rad(ref_lon)
        lat_w = lat_u 
        lon_w = lon_v 

        # Haversine formula
        dlon_u_w = lon_w - lon_u

        hav_theta_u_w = np.cos(lat_u) * np.cos(lat_w) * np.sin(dlon_u_w/ 2)**2

        # Distance in meters
        b = 2*R*np.arcsin(np.sqrt(hav_theta_u_w))
        a = np.abs(lat_v-lat_w)*R

        return b,a

def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    # https://en.wikipedia.org/wiki/Haversine_formula 
    # Look in Formulation for equation
    
    # Earth radius in meters
    R = 6378137.0

    # Convert degrees to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    ref_lat = np.deg2rad(ref_lat)
    ref_lon = np.deg2rad(ref_lon)

    # Haversine formula
    dlat = lat - ref_lat
    dlon = lon - ref_lon

    hav_theta = np.sin(dlat / 2)**2 + np.cos(ref_lat) * np.cos(lat) * np.sin(dlon / 2)**2

    # Distance in meters
    distance = 2*R*np.arcsin(np.sqrt(hav_theta))

    return distance

def convert_gps_to_meters(longitudes, latitudes):
    # Determine the reference point (minimum latitude and longitude)
    ref_lat = np.min(latitudes)
    ref_lon = np.min(longitudes)

    # Calculate distances
    # For x_meters, latitude is constant (ref_lat) and longitude varies
    x_meters = latlon_to_meters(np.full_like(longitudes, ref_lat), longitudes, ref_lat, ref_lon)
    # For y_meters, longitude is constant (ref_lon) and latitude varies
    y_meters = latlon_to_meters(latitudes, np.full_like(latitudes, ref_lon), ref_lat, ref_lon)

    return x_meters, y_meters





    




 


    


