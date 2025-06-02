import cv2
import math
import numpy as np
import cuda_akaze_py as akaze

def test_cpu_vs_cuda_akaze(image_path1, image_path2, num_matches=30):
    # Read images grayscale
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # --- CPU AKAZE ---
    cpu_detector = cv2.AKAZE_create()
    kp1_cpu, des1_cpu = cpu_detector.detectAndCompute(img1, None)
    kp2_cpu, des2_cpu = cpu_detector.detectAndCompute(img2, None)

    bf_cpu = cv2.BFMatcher(cv2.NORM_HAMMING if des1_cpu is not None and des1_cpu.dtype == np.uint8 else cv2.NORM_L2, crossCheck=True)
    matches_cpu = bf_cpu.match(des1_cpu, des2_cpu) if des1_cpu is not None and des2_cpu is not None else []
    matches_cpu = sorted(matches_cpu, key=lambda x: x.distance)[:num_matches]

    print(f"CPU AKAZE detected {len(kp1_cpu)} and {len(kp2_cpu)} keypoints")
    print(f"CPU AKAZE matches: {len(matches_cpu)}")

    for i, match in enumerate(matches_cpu):
        print(f"Match {i+1}:")
        print(f"  Query Index: {match.queryIdx}")
        print(f"  Train Index: {match.trainIdx}")
        print(f"  Distance: {match.distance}")
        print("-----------------------------")

    # --- CUDA AKAZE ---
    img1_32 = np.float32(img1) / 255.0
    img2_32 = np.float32(img2) / 255.0

    options = akaze.AKAZEOptions()
    options.setWidth(img1.shape[1])
    options.setHeight(img1.shape[0])
    evo1 = akaze.AKAZE(options)
    evo1.Create_Nonlinear_Scale_Space(img1_32)
    des1_cuda, kp1_cuda_raw = evo1.Compute_Descriptors()

    options.setWidth(img2.shape[1])
    options.setHeight(img2.shape[0])
    evo2 = akaze.AKAZE(options)
    evo2.Create_Nonlinear_Scale_Space(img2_32)
    des2_cuda, kp2_cuda_raw = evo2.Compute_Descriptors()

    print(f"CUDA AKAZE des1 shape: {des1_cuda.shape}, dtype: {des1_cuda.dtype}")
    print(f"CUDA AKAZE des2 shape: {des2_cuda.shape}, dtype: {des2_cuda.dtype}")
    print(f"CUDA AKAZE raw keypoints count: {len(kp1_cuda_raw)}, {len(kp2_cuda_raw)}")

#    # Debugging: Check first few descriptors
    print("First few CUDA descriptors (raw, uint8):")
    print(des1_cuda[:5])
    print(des2_cuda[:5])
    
    print("First few CPU descriptors (raw, uint8):")
    print(des1_cpu[:5])
    print(des2_cpu[:5])

    # Convert raw kp to cv2.KeyPoint objects
    kp1_cuda = [
        cv2.KeyPoint(
            float(pt[0]) if not math.isnan(pt[0]) and not math.isinf(pt[0]) else 0.0,  # x coordinate
            float(pt[1]) if not math.isnan(pt[1]) and not math.isinf(pt[1]) else 0.0,  # y coordinate
            float(pt[2]) if not math.isnan(pt[2]) and not math.isinf(pt[2]) else 0.0,  # size
            float(pt[3]) if not math.isnan(pt[3]) and not math.isinf(pt[3]) else 0.0,  # angle
            float(pt[4]) if not math.isnan(pt[4]) and not math.isinf(pt[4]) else 0.0,  # response
            int(pt[5]) if not math.isnan(pt[5]) and not math.isinf(pt[5]) else 0,      # octave (or default value)
            int(pt[6]) if not math.isnan(pt[6]) and not math.isinf(pt[6]) else 0       # class_id (or default value)
        )
        for pt in kp1_cuda_raw
    ]

    kp2_cuda = [
            cv2.KeyPoint(
            float(pt[0]) if not math.isnan(pt[0]) and not math.isinf(pt[0]) else 0.0,  # x coordinate
            float(pt[1]) if not math.isnan(pt[1]) and not math.isinf(pt[1]) else 0.0,  # y coordinate
            float(pt[2]) if not math.isnan(pt[2]) and not math.isinf(pt[2]) else 0.0,  # size
            float(pt[3]) if not math.isnan(pt[3]) and not math.isinf(pt[3]) else 0.0,  # angle
            float(pt[4]) if not math.isnan(pt[4]) and not math.isinf(pt[4]) else 0.0,  # response
            int(pt[5]) if not math.isnan(pt[5]) and not math.isinf(pt[5]) else 0,      # octave (or default value)
            int(pt[6]) if not math.isnan(pt[6]) and not math.isinf(pt[6]) else 0       # class_id (or default value)
        )
        for pt in kp2_cuda_raw
    ]

    # If you want to compare each attribute individually
    for I in range(0,len(kp2_cpu)):
        keypoint_1 = kp2_cpu[I]
        print("KeyPoint kp1 CPU:", keypoint_1.pt, keypoint_1.size, keypoint_1.angle, keypoint_1.response,  keypoint_1.octave,  keypoint_1.class_id)
    
    for I in range(0,len(kp2_cuda)):
        keypoint_2 = kp2_cuda[I]
        print("KeyPoint kp1 CUDA:", keypoint_2.pt, keypoint_2.size, keypoint_2.angle, keypoint_2.response, keypoint_2.octave, keypoint_2.class_id)

    # Normalize CUDA descriptors if needed (for float32 descriptors)
    if des1_cuda.dtype == np.uint8:
        des1_cuda = des1_cuda.astype(np.float32) / 255.0  # Normalize uint8 descriptors
        des2_cuda = des2_cuda.astype(np.float32) / 255.0  # Normalize uint8 descriptors
    else:
        des1_cuda = cv2.normalize(des1_cuda.astype(np.float32), None, 0, 1, cv2.NORM_L2)
        des2_cuda = cv2.normalize(des2_cuda.astype(np.float32), None, 0, 1, cv2.NORM_L2)

    # Use Hamming distance if descriptors are uint8, otherwise L2 distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if des1_cuda.dtype == np.uint8 else cv2.NORM_L2, crossCheck=True)
    matches_cuda = bf.match(des1_cuda, des2_cuda)

    #bf = akaze.Matcher()
    #matches_cuda = bf.BFMatch(des1_cuda, des2_cuda)

    print(f"Filtered CUDA matches count: {len(matches_cuda)}")

    # Draw matches and save images
    match_img_cpu = cv2.drawMatches(img1, kp1_cpu, img2, kp2_cpu, matches_cpu, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matches_cpu.png", match_img_cpu)

    match_img_cuda = cv2.drawMatches(img1, kp1_cuda, img2, kp2_cuda, matches_cuda, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matches_cuda.png", match_img_cuda)

    print("Match images saved as matches_cpu.png and matches_cuda.png")


if __name__ == "__main__":
    # Replace these with paths to two test images you want to compare
    test_cpu_vs_cuda_akaze("./datasets-map/img1.jpg", "./datasets-map/img7.jpg")

