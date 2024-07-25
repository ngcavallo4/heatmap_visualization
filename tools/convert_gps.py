import numpy as np

def calculate(lat, lon, ref_lat, ref_lon):

    R = 6378100 # Radius of Earth in m 

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
    
    e = np.arcsin((np.sin(E) * np.sin(d)) / np.sin(D))

    long_m = e 
    lat_m = d 

    return long_m, lat_m 

def convert_coordinates_to_meters(lats, lons, ref_lat, ref_lon):
    long_m_array = []
    lat_m_array = []
    
    for lat, lon in zip(lats, lons):
        long_m, lat_m = calculate(lat, lon, ref_lat, ref_lon)
        long_m_array.append(long_m)
        lat_m_array.append(lat_m)
    
    return np.array(long_m_array), np.array(lat_m_array)




    