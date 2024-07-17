import csv
import os
import numpy as np

#
##
##
###
### You must be in the /kriging folder for this to work!
PATH = "./data"

class CSVParser():

    r"""Utility class that parses logging CSVs. Separates data into X position, 
       Y position, and stiffness value by leg and for all 4 legs to enable
       plotting by each leg and plotting for entire traversal.
        Parameters
        ----------
        filename: :class:`str`
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

    def __init__(self, filename: str):

        self.data_dict = {}
        self.data_arr = []

        filepath = os.path.join(PATH, filename) 

        self.extract_data(filepath)

    def extract_data(self, filepath: os.PathLike):
        r"""Separates data into 2 data structures, a dictionary of arrays for
            singular leg data, and an array of the combined data. 
            Parameters
            ----------
            filepath: :class:`os.PathLike`
                Filepath to csv file.
        """

        self.data_dict = {0: [], 1: [], 2: [], 3: []}
        self.data_arr = []

        with open(filepath, 'r') as csvfile:
            filereader = csv.reader(csvfile,delimiter=",")
            next(filereader) # Skips title row 

            for row in filereader:
                row_value = int(row[5]) 
                self.data_dict[row_value].append([float(row[1]), float(row[2]), 
                                                  float(row[3])])

                self.data_arr.append([float(row[1]), float(row[2]), float(row[3])])  

        self.data_arr = np.array(self.data_arr)

    def access_data(self, request: str):

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

        request = str(request)
        match request:
            case '0':
                leg_0 = np.array(self.data_dict[0])

                leg0_x = leg_0[:,0]
                leg0_y = leg_0[:,1]
                leg0_stiff = leg_0[:,2]
                "Front Left"

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
            case 'all':
                all_legs_x = self.data_arr[:,0]
                all_legs_y = self.data_arr[:,1]
                all_legs_stiff = self.data_arr[:,2]
                title = "All Legs"

                return all_legs_x, all_legs_y, all_legs_stiff, title
