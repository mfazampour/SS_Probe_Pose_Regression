# import numpy as np
# from scipy import misc
# from scipy.interpolate import interp2d
# from math import pi, atan2, hypot
# import cv2

# inputImagePath = '/home/aorta-scan/dani/repos/cactuss_end2end/results_test/intensity_map.png'
# resultWidth = 800
# resultHeight = 600
# centerX = resultWidth / 2
# centerY = - 50.0
# maxAngle =  45.0 / 2 / 180 * pi
# minAngle = -maxAngle
# minRadius = 100.0
# maxRadius = 600.0

# inputImage = np.swapaxes(cv2.imread(inputImagePath, cv2.IMREAD_GRAYSCALE),1,0)
# # inputImage = np.swapaxes(cv2.imread(inputImagePath),1,0)
# h,w = inputImage.shape
# print("h = {h} w = {w} }")
# channels = inputImage[:,:]
# interpolated = [interp2d(range(w), range(h), c) for c in channels]
# resultImage = np.zeros([resultHeight, resultWidth], dtype = np.uint8)

# for c in range(resultWidth):
#   for r in range(resultHeight):
#     dx = c - centerX
#     dy = r - centerY
#     angle = atan2(dx, dy) # yes, dx, dy in this case!
#     if angle < maxAngle and angle > minAngle:
#       origCol = (angle - minAngle) / (maxAngle - minAngle) * w
#       radius = hypot(dx, dy)
#       if radius > minRadius and radius < maxRadius:
#         origRow = (radius - minRadius) / (maxRadius - minRadius) * h
#         resultImage[r, c] = interpolated(origCol, origRow)

# import matplotlib.pyplot as plt
# plt.imshow(resultImage)
# plt.show()


import numpy as np
from scipy import misc
from scipy.interpolate import interp2d
from math import pi, atan2, hypot
import cv2
from scipy.interpolate import griddata


inputImagePath =  '/home/aorta-scan/dani/repos/cactuss_end2end/results_test/intensity_map.png'
resultWidth = 800
resultHeight = 600
centerX = resultWidth / 2
centerY = - 50.0
maxAngle =  45.0 / 2 / 180 * pi
minAngle = -maxAngle
minRadius = 100.0
maxRadius = 600.0

inputImage = np.swapaxes(cv2.imread(inputImagePath, cv2.IMREAD_GRAYSCALE),1,0)
h,w = inputImage.shape
print(f"h = {h} w = {w}")

# interpolated = interp2d(range(w), range(h), inputImage)
# resultImage = np.zeros([resultHeight, resultWidth], dtype = np.uint8)

# for c in range(resultWidth):
#   for r in range(resultHeight):
#     dx = c - centerX
#     dy = r - centerY
#     angle = atan2(dx, dy) # yes, dx, dy in this case!
#     if angle < maxAngle and angle > minAngle:
#       origCol = (angle - minAngle) / (maxAngle - minAngle) * w
#       radius = hypot(dx, dy)
#       if radius > minRadius and radius < maxRadius:
#         origRow = (radius - minRadius) / (maxRadius - minRadius) * h
#         resultImage[r, c] = interpolated(origCol, origRow)

# import matplotlib.pyplot as plt
# plt.imshow(resultImage, cmap='gray')
# plt.show()


# Create arrays of x and y coordinates for each pixel in the result image
x = np.arange(resultWidth)
y = np.arange(resultHeight)
xx, yy = np.meshgrid(x, y)

# Calculate the angle and radius for each pixel
dx = xx - centerX
dy = yy - centerY
angle = np.arctan2(dx, dy)
radius = np.hypot(dx, dy)

# Create a boolean mask to select the pixels that meet the criteria for angle and radius
angle_mask = (angle < maxAngle) & (angle > minAngle)
radius_mask = (radius > minRadius) & (radius < maxRadius)
mask = angle_mask & radius_mask

# # Create arrays of the original row and column indices for the selected pixels
# orig_col = (angle[mask] - minAngle) / (maxAngle - minAngle) * w
# orig_row = (radius[mask] - minRadius) / (maxRadius - minRadius) * h

# # Interpolate the pixel values for the selected pixels
# interpolated = griddata((np.arange(w), np.arange(h)), inputImage.ravel(), (np.unique(orig_col), np.unique(orig_row)), method='linear')

# # Reshape the interpolated values into the shape of the result image
# resultImage = np.zeros([resultHeight, resultWidth], dtype=np.uint8)
# resultImage[mask] = np.round(interpolated).astype(np.uint8)

# Compute the original column and row indices for the selected pixels
orig_col = (angle - minAngle) / (maxAngle - minAngle) * (w - 1)
orig_row = (radius - minRadius) / (maxRadius - minRadius) * (h - 1)

# Filter out invalid indices
mask = ((orig_col >= 0) & (orig_col < w) & (orig_row >= 0) & (orig_row < h))
orig_col = orig_col[mask]
orig_row = orig_row[mask]

# Interpolate the pixel values for the selected pixels
interpolated = griddata((np.arange(w), np.arange(h)), inputImage.ravel(), (orig_col, orig_row), method='linear')

# Reshape the interpolated values into the shape of the result image
resultImage[r, c] = interpolated.reshape((-1, 3))
