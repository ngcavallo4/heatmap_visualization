import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from shipeng_methods import Gaussian_Estimation
from src.utility.parse_csv import CSVParser

def organize_area(x,y, match_steps, x_interpolation_input_range = None, y_interpolation_input_range = None):

    x_interpolation_range = [0,0]
    y_interpolation_range = [0,0]

    if match_steps:
                # If values are given, set those equal
                if x_interpolation_input_range is not None and y_interpolation_input_range is not None:
                    x_interpolation_range[0] = x_interpolation_input_range[0]
                    y_interpolation_range[0]= y_interpolation_input_range[0]
                else: # Else set values equal to minimum of steps 
                    x_interpolation_range[0] = np.min(x) - 0.1
                    y_interpolation_range[0] = np.min(y) - 0.1

                # Set upper range equal to maximum of steps
                x_interpolation_range[1] = np.max(x) + 0.1
                y_interpolation_range[1] = np.max(y) + 0.1
    else: # If not match steps, then must pass in values
        if x_interpolation_input_range is None or y_interpolation_input_range is None:
            raise BaseException('Missing arguments. If match_steps is false, the four other arguments in organize_kriging_area are required.')

        x_interpolation_range[0] = x_interpolation_input_range[0]
        x_interpolation_range[1] = x_interpolation_input_range[1]
        y_interpolation_range[0]= y_interpolation_input_range[0]
        y_interpolation_range[1]= y_interpolation_input_range[1]

    return x_interpolation_range, y_interpolation_range

parser = CSVParser('log00-19_trans.csv')
x, y, stiffness, title = parser.access_data('all')
x_range, y_range = organize_area(x,y,True)

# using gaussian process to predict
estimatedNum = 100
xx1, xx2 = np.linspace(x_range[0], x_range[1], num=estimatedNum), np.linspace(y_range[0], y_range[1], num=estimatedNum)
vals = np.array([[x1_, x2_] for x1_ in xx1 for x2_ in xx2])

robot_measured_points = np.array([x, y]).T
shear_prediction, shear_std, information_shear = Gaussian_Estimation(robot_measured_points,  stiffness,   vals, False, 0.2, 0.15, 4)

shear_prediction = shear_prediction.reshape(estimatedNum, estimatedNum)
information_shear = information_shear.reshape(estimatedNum, estimatedNum)
shear_std = shear_std.reshape(estimatedNum, estimatedNum)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

Information_image = ax[0].imshow(shear_std, cmap='viridis',extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
ax[0].set_title('Uncertainty (Std) Map')
cb_uncertainty = fig.colorbar(Information_image, ax=ax[0], label='uncertainty')

shear_strength_image = ax[1].imshow(shear_prediction, cmap='viridis', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
ax[1].scatter(x,y,c=stiffness,cmap='viridis', edgecolors='k')
cb_strength = fig.colorbar(shear_strength_image, ax=ax[1], label='Shear Strength')
ax[1].set_title('Shear Strength Predicted')
plt.show()