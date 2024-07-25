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

def calculate(lat, lon, ref_lat, ref_lon):

    R = 6378137.0 # Radius of Earth in m 

    # Convert degrees to radians using NumPy
    lat = np.deg2rad(lat)
    ref_lat = np.deg2rad(ref_lat)
    lon = np.deg2rad(lon)
    ref_lon = np.deg2rad(ref_lon)
    
    # Calculate differences
    delta_lat_w_v = lat - ref_lat
    delta_lon_w_v = lon - ref_lon
    delta_lat_u_v = np.pi / 2 - lat  # lat_u is 90 degrees (north pole)
    delta_lat_u_w = np.pi / 2 - ref_lat

    # Calculate haversine of theta
    hav_theta = np.sin(delta_lat_w_v / 2)**2 + np.cos(ref_lat) * np.cos(lat) * np.sin(delta_lon_w_v / 2)**2
    
    # Calculate c using the haversine formula
    c = 2 * R * np.arcsin(np.sqrt(hav_theta))
    
    # Additional calculations as per the provided formulas
    a = R * delta_lat_u_v
    b = delta_lat_u_w * R
    d = delta_lat_w_v * R

    # Spherical law of cosines to calculate A and B
    A = np.arcsin((np.sin(a) * np.sin(c)) / np.sin(c)) 
    B = np.arcsin((np.sin(b) * np.sin(c)) / np.sin(c))

    # Using constructed geometry to find angles D and E 
    D = np.pi / 2 - B
    E = np.pi - A

    # cos(c) = cos(a)*cos(b) + sin(a)*sin(b)*cos(C)
    
    # cos(e) = cos(c)*cos(d) + sin(c)*sin(d)*cos(E) 

    temp = np.cos(c)*np.cos(d) + np.sin(c)*np.sin(d)*np.cos(E)
    e = np.arccos(temp)

    # NOTE: SOMETHING IS GOING WRONG HERE......... 
    # temp = (np.sin(E) * np.sin(d)) / np.sin(D)
    # if temp > 1:
    #     temp = 1
    # e = np.arcsin(temp)

    long_m = np.abs(e)
    lat_m = np.abs(d)

    if np.isnan(long_m) or np.isnan(lat_m):
        raise ValueError("NaN Arg Detected")

    return long_m, lat_m 





    




 


    


