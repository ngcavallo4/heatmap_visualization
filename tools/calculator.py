small_scale = 0.03/1111110 # deg/m * m = deg 
med_scale = 0.1/1111110 # deg/m * m = deg
long_scale = 0.5/1111110 # deg/m * m = deg 

# print(4.5000045000045006e-07*1111110) # deg * 11111100 m/deg = m
import numpy as np

def rotate_matrix(matrix, theta):

    match theta:
        case 90:
            return np.flipud(np.transpose(matrix))
        case 180:
            return np.flipud(np.fliplr(matrix))
        case 270:
            return np.fliplr(np.transpose(matrix))

# Example usage
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

rotated_90 = rotate_matrix(matrix, 90)
rotated_180 = rotate_matrix(matrix, 180)
rotated_270 = rotate_matrix(matrix, 270)

print("Original matrix:")
print(matrix)

print("Rotated matrix by 90 degrees clockwise:")
print(rotated_90)

print("Rotated matrix by 180 degrees:")
print(rotated_180)

print("Rotated matrix by 270 degrees clockwise:")
print(rotated_270)
