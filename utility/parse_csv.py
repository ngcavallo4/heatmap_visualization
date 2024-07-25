import csv
import os
import numpy as np
import utility.convert_gps as gpsconvert
import pandas 

#
##
##
###
### You must be in the /heatmap_visualization folder for this to work!
PATH = "./data"

class CSVParser():

    r"""Utility class that parses logging CSVs. Separates data into X position, 
       Y position, and stiffness value by leg and for all 4 legs to enable
       plotting by each leg and plotting for entire traversal.

        Parameters
        ----------
        filepath: :class:`str`
            Name of the csv file where the data is stored. This file should be
            stored in /kriging/data. This will only work if you are in /kriging
            when running your code.
        data_dict: :class:`dict` 
            Dictionary where data of individual legs is stored. Access it using
            the index of the leg.For each leg, x-position is stored in column 0,
            y-position is stored in column 1, and stiffness is stored in column 2. 
        data_arr: :class:`list`
            Array where data of entire traversal is stored. X position is stored
            in column 0, y position is stored in column 1, stiffness is stored in
            column 2.
    """

    def __init__(self, filepath: str):

        self.data_dict = {}
        self.data_arr = []
        self.filepath = filepath
        
        # filepath = os.path.join(PATH, filename) 

        self.extract_data()

    def extract_data(self):
        r"""Separates data into 2 data structures, a dictionary of arrays for
            singular leg data, and an array of the combined data. Converts
            GPS coordinates from degrees to microdegrees (x 1 million), 
            because the models really don't like how small the footsteps
            are in terms of degrees. 

            Parameters
            ----------

            filepath: :class:`os.PathLike`
                Filepath to csv file.
        """

        self.data_dict = {0: [], 1: [], 2: [], 3: []}
        self.data_arr = []

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
                
                self.data_dict[row_value].append([float(row[2]),float(row[1]),float(row[3])])

        self.data_arr = np.array(self.data_arr)

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
            x: :class:`numpy.ndarray`
                X position array.
            y: :class:`numpy.ndarray`
                Y position array.
            stiffness: :class:`numpy.ndarray`
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

                    # leg0_x_meters, leg0_y_meters = gpsconvert.gps_to_meters(leg0_y, leg0_x)
                    # return leg0_x_meters, leg0_y_meters, leg0_stiff, title
                    
                    return leg0_x, leg0_y, leg0_stiff, title

                case '1':
                    leg_1 = np.array(self.data_dict[1])

                    leg1_x = leg_1[:, 0]
                    leg1_y = leg_1[:, 1]
                    leg1_stiff = leg_1[:, 2]
                    title = "Back Left"

                    # # leg1_x_meters, leg1_y_meters = gpsconvert.gps_to_meters(leg1_y, leg1_x)
                    # return leg1_x_meters, leg1_y_meters, leg1_stiff, title

                    return leg1_x, leg1_y, leg1_stiff, title
                
                case '2':
                    leg_2 = np.array(self.data_dict[2])

                    leg2_x = leg_2[:, 0]
                    leg2_y = leg_2[:, 1]
                    leg2_stiff = leg_2[:, 2]
                    title = "Front Right"

                    # leg2_x_meters, leg2_y_meters = gpsconvert.gps_to_meters(leg2_y, leg2_x)
                    # return leg2_x_meters, leg2_y_meters, leg2_stiff, title
                
                    return leg2_x, leg2_y, leg2_stiff, title

                case '3':
                    leg_3 = np.array(self.data_dict[3])

                    leg3_x = leg_3[:, 0]
                    leg3_y = leg_3[:, 1]
                    leg3_stiff = leg_3[:, 2]
                    title = "Back Right"

                    # leg3_x_meters, leg3_y_meters = gpsconvert.gps_to_meters(leg3_y, leg3_x)
                    # return leg3_x_meters, leg3_y_meters, leg3_stiff, title
                
                    return leg3_x, leg3_y, leg3_stiff, title


