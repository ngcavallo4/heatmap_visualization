import numpy as np

def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    # https://en.wikipedia.org/wiki/Haversine_formula 
    # Look in Formulation for equation
    
    # Earth radius in meters
    R = 6378137.0

    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    ref_lat = np.radians(ref_lat)
    ref_lon = np.radians(ref_lon)

    # Haversine formula
    dlat = lat - ref_lat
    dlon = lon - ref_lon

    hav_theta = np.sin(dlat / 2)**2 + np.cos(ref_lat) * np.cos(lat) * np.sin(dlon / 2)**2

    # Distance in meters
    distance = 2*R*np.arcsin(np.sqrt(hav_theta))

    return distance

def gps_to_meters(latitudes, longitudes):
    # Determine the reference point (minimum latitude and longitude)
    ref_lat = np.min(latitudes) - 0.000002
    ref_lon = np.min(longitudes) - 0.000002

    x_meters = []
    y_meters = []

    # Calculate distances
    for i in len(latitudes):
        lat = latitudes[i]
        lon = longitudes[i]
        # d = latlon_to_meters(lat,lon,ref_lat,ref_lon)

        x_met, y_met = law_of_sines(ref_lat, ref_lon, lat, lon)
        x_meters.append(x_met)
        y_met.append(y_met)

    x_meters = np.array(x_meters)
    y_meters = np.array(y_meters)

    return x_meters, y_meters

def law_of_sines(ref_lat: float,ref_lon: float,lat: float, lon: float):

    R = 6378137.0
    
    alpha = np.deg2rad(90 - ref_lat) # central angle defining arc a 
    a = alpha*R

    beta = np.deg2rad(90 - lat) # central angle defining arc b
    b = beta*R 

    delta = np.deg2rad(lat - ref_lat) # sin^2(dlat_w_v/2) is even function so doesn't matter order
    C = np.deg2rad(lon - ref_lon) # difference in longitudes, also angle opposite side c (close to north pole)

    # haversine function = sin(theta/2)**2

    hav_theta = np.sin(delta/2)**2 + np.cos(ref_lat)*np.cos(lat)*np.sin(C/2)**2 # haversine formula
    c = 2*R*np.arcsin(np.sqrt(hav_theta)) # archaversine formula

    # solving for angles A and B using spherical law of sines
    A = np.arcsin((np.sin(a)*np.sin(C))/np.sin(c))
    B = np.arcsin((np.sin(b)*np.sin(C))/np.sin(c))

    D = np.pi/2 - B # D is along a line of latitude and makes a right angle at reference pt
    E = np.pi - A # A makes a straight line with E

    d = delta*R # since d is along a line of latitude, we can use the difference in latitude directly 

    # solving for e using spherical law of sines
    e = np.arcsin((np.sin(E)*np.sin(d))/np.sin(D)) 

    long_m = e 
    lat_m = d 

    return long_m, lat_m 

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
    
    # NOTE: SOMETHING IS GOING WRONG HERE......... 
    e = np.arcsin((np.sin(E) * np.sin(d)) / np.sin(D))

    long_m = np.abs(e)
    lat_m = np.abs(d)

    if np.isnan(long_m) or np.isnan(lat_m):
        raise ValueError("NaN Arg Detecte")

    return long_m, lat_m 

def gps_coords_to_meters(lons, lats):

    ref_lat = np.min(lats)
    ref_lon = np.min(lons)

    long_m_array = []
    lat_m_array = []
    
    for lat, lon in zip(lats, lons):
        long_m, lat_m = calculate(lat, lon, ref_lat, ref_lon)
        long_m_array.append(long_m)
        lat_m_array.append(lat_m)

    
    return np.array(long_m_array), np.array(lat_m_array)




    




 


    


