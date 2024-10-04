import csv
import os
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
                # self.data_dict[row_value].append([float(row[1])*1000000,
                #                                 float(row[2])*1000000, 
                #                                 float(row[3])])
                
                # row 1 = lat, row 2 = lon, row 3 = stiff
                self.data_dict[row_value].append([float(row[1]),float(row[2]),float(row[3])])

    def convert_gps_to_meters(self):

        with open(self.filepath, 'a') as csvfile:
            filewriter = csv.writer(csvfile)
            filereader = csv.reader(csvfile)
            next(filereader)


    def access_data(self, request: list[str]):

        r"""Enables access to the correct entries of the dictionary of arrays
            for single leg plotting and combined leg plotting. 

            Parameters
            ----------
            request: :class:`str`
                The leg/legs that are being requested. If all legs are being
                requested, enter 'all'. 
            
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

        for s in request:

            match s:
                case '0':
                    leg_0 = np.array(self.data_dict[0])

                    leg0_x = leg_0[:,0]
                    leg0_y = leg_0[:,1]
                    leg0_stiff = leg_0[:,2]
                    title = "Front Left"
                    
                    return leg0_x, leg0_y, leg0_stiff, title

                case '1':
                    leg_1 = np.array(self.data_dict[1])
                    leg1_x = leg_1[:, 0]
                    leg1_y = leg_1[:, 1]
                    leg1_stiff = leg_1[:, 2]
                    title = "Back Left"

                    return leg1_x, leg1_y, leg1_stiff, title
                
                case '2':
                    leg_2 = np.array(self.data_dict[2])

                    leg2_x = leg_2[:, 0]
                    leg2_y = leg_2[:, 1]
                    leg2_stiff = leg_2[:, 2]
                    title = "Front Right"
                
                    return leg2_x, leg2_y, leg2_stiff, title

                case '3':
                    leg_3 = np.array(self.data_dict[3])

                    leg3_x = leg_3[:, 0]
                    leg3_y = leg_3[:, 1]
                    leg3_stiff = leg_3[:, 2]
                    title = "Back Right"
                
                    return leg3_x, leg3_y, leg3_stiff, title


