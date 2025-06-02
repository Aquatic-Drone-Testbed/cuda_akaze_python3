import sys
import os
import numpy as np
import cv2

# Ensure the current directory (or the directory with the .so file) is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the pybind11 module
import cuda_akaze_py as akaze

# Load grayscale images
img1 = cv2.imread("./datasets-map/img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./datasets-map/img7.jpg", cv2.IMREAD_GRAYSCALE)

# Convert images to float32 in range [0, 1]
img1_32 = np.float32(img1) / 255.0
img2_32 = np.float32(img2) / 255.0

# Set AKAZE options
options = akaze.AKAZEOptions()

# First image
options.setWidth(img1.shape[1])
options.setHeight(img1.shape[0])
evo1 = akaze.AKAZE(options)
evo1.Create_Nonlinear_Scale_Space(img1_32)
desc1, kpts1 = evo1.Compute_Descriptors()

# Second image
options.setWidth(img2.shape[1])
options.setHeight(img2.shape[0])
evo2 = akaze.AKAZE(options)
evo2.Create_Nonlinear_Scale_Space(img2_32)
desc2, kpts2 = evo2.Compute_Descriptors()

for pt in kpts1:
    print("==")
    print(pt[0])
    print(pt[1])
    print(pt[2])
    print(pt[3])
    print(pt[4])
    print(pt[5])
    print(pt[6])

# Match descriptors
matcher = akaze.Matcher()
matches = matcher.BFMatch(desc1, desc2)

# Print results
print(f"Detected {len(kpts1)} and {len(kpts2)} keypoints.")
print(f"Found {len(matches)} matches.")

print(f"{matches}")

