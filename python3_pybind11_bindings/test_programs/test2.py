import cv2
import numpy as np
import time
import sys
from cuda_akaze_py import AKAZEOptions, AKAZE, Matcher

# Image matching constants
MIN_H_ERROR = 2.50  # Maximum error in pixels to accept an inlier
DRATIO = 0.80       # NNDR Matching value

# Parse the input arguments (similar to the C++ parse_input_options)
def parse_input_options():
    if len(sys.argv) < 3:
        print("Usage: python3 akaze_match.py <image_path1> <image_path2>")
        sys.exit(1)

    img_path1 = sys.argv[1]
    img_path2 = sys.argv[2]
    return img_path1, img_path2

def main():
    img_path1, img_path2 = parse_input_options()

    # Load images
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Error loading images: {img_path1}, {img_path2}")
        sys.exit(1)

    # Convert images to float (normalized between 0 and 1)
    img1_32 = img1.astype(np.float32) / 255.0
    img2_32 = img2.astype(np.float32) / 255.0

    # Create AKAZE options and initialize the AKAZE objects
    options = AKAZEOptions()
    options.setWidth(img1.shape[1])
    options.setHeight(img1.shape[0])

    akaze1 = AKAZE(options)
    akaze2 = AKAZE(options)

    # Start the timer
    start_time = time.time()

    # Perform feature detection and descriptor computation
    akaze1.Create_Nonlinear_Scale_Space(img1_32)
    kpts1, desc1 = akaze1.Feature_Detection()
    akaze1.Compute_Descriptors(kpts1, desc1)

    akaze2.Create_Nonlinear_Scale_Space(img2_32)
    kpts2, desc2 = akaze2.Feature_Detection()
    akaze2.Compute_Descriptors(kpts2, desc2)

    akaze_time = time.time() - start_time

    # Perform feature matching using Brute Force Matcher
    start_time = time.time()

    matcher = Matcher()
    matches = matcher.BFMatch(desc1, desc2)

    match_time = time.time() - start_time

    # Convert matches to NumPy arrays
    matches = matches.reshape(-1, 2)

    # Display the results
    print(f"Number of Keypoints Image 1: {len(kpts1)}")
    print(f"Number of Keypoints Image 2: {len(kpts2)}")
    print(f"A-KAZE Features Extraction Time (s): {akaze_time:.3f}")
    print(f"Matching Descriptors Time (s): {match_time:.3f}")
    print(f"Number of Matches: {matches.shape[0]}")

    # Visualize matching (Optional)
    img_matches = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

