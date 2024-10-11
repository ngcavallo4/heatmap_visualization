import csv
import os
import re
import numpy as np

class CSVParser():

    r"""Utility class that parses logging CSVs. Separates data into X position, 
       Y position, and stiffness value by leg and for all 4 legs to enable
       plotting by each leg and plotting for entire traversal.

        Parameters
        ----------
        path: :class:`os.PathLike`
            Path to the folder containing CSV files. 
        filepath: :class:`str`
            Full path of the csv file where the data is stored. 
        data_dict: :class:`dict` 
            Dictionary where data of individual legs is stored. Access it using
            the index of the leg. For each leg, x-position is stored in column 0,
            y-position is stored in column 1, and stiffness is stored in column 2. 
    """

    def __init__(self, path, filename: str):
        """Initialize the CSV class with the given file.
        
        Parameters
        ----------
        filename: :class:`str`
            Filename to parse.
        """

        self.data_dict = {}
        self.filepath = os.path.join(path, filename) 

        self.extract_data()

    def extract_data(self):
        r"""Separates data into 2 data structures, a dictionary of arrays for
            singular leg data, and an array of the combined data. Converts
            GPS coordinates from degrees to microdegrees (x 1 million), 
            because the models really don't like how small the footsteps
            are in terms of degrees. 
        """

        self.data_dict = {0: [], 1: [], 2: [], 3: []}

        with open(self.filepath, 'r') as csvfile:
            filereader = csv.reader(csvfile,delimiter=",")
            next(filereader) # Skips title row 

            # Parses each row and separates them by leg index into data_dict,
            # and into data_arr all togther regardless of leg index. 

            # Converting degrees to microdegrees
            for row in filereader:
                row_value = int(row[5]) 
                # row 1 = lat, row 2 = lon, row 3 = stiff
                self.data_dict[row_value].append([float(row[1]),float(row[2]),float(row[3])])

    def convert_gps_to_meters(self):

        with open(self.filepath, 'a') as csvfile:
            filewriter = csv.writer(csvfile)
            filereader = csv.reader(csvfile)
            next(filereader)


    def access_data(self, request: list[str]|str):

        r"""Enables access to the correct entries of the dictionary of arrays
            for single leg plotting and combined leg plotting. 

            Parameters
            ----------
            request: :class:`str`
                The leg/legs that are being requested.
            
            Returns
            -------
            x: :class:`np.ndarray`
                X position array.
            y: :class:`np.ndarray`
                Y position array.
            stiffness: :class:`np.ndarray`
                Stiffness value array.
            title: :class:`str`
                Title of request, for plotting purposes.

        """
        if isinstance(request,list):
            request=request[0]
        for req in request:
            try:
                float(req)
            except ValueError as e:
                print("Requests must contain valid indices")
                raise e
        
        leg_data = []
        titles = []
        LEG_OPTIONS = {"0":(0,"Front Left"),
                       "1":(1,"Back Left"),
                       "2":(2,"Front Right"),
                       "3":(3,"Back Right")}
        
        for leg,leg_info in LEG_OPTIONS.items():
            if leg in request:
                leg_data.extend(self.data_dict[leg_info[0]])
                titles.append(leg_info[1])
        title = '/'.join(titles)
        
        leg_data = np.array(leg_data)

        leg_data_x = leg_data[:,0]
        leg_data_y = leg_data[:,1]
        leg_data_stiff = leg_data[:,2]
        
        return leg_data_x, leg_data_y, leg_data_stiff, title