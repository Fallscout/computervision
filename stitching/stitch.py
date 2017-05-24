import numpy as np
import cv2

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="")
    parser.add_argument("img1", type=str, help="")
    parser.add_argument("img2", type=str, help="")
    args = parser.parse_args()
    
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)

    output_image = im_stitch(img1, img2)

def im_stitch(img1, img2):
    gray_img1 = np.float32(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    gray_img2 = np.float32(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

    keypoints1 = detect_corners(gray_img1, 2, 3, 0.04, 0.01)
    keypoints2 = detect_corners(gray_img2, 2, 3, 0.04, 0.01)

    patches1 = extract_patches(gray_img1, keypoints1, 3, 3)
    patches2 = extract_patches(gray_img2, keypoints2, 3, 3)

    distances = calculate_distances(patches1, patches2)

    corr_indices = np.argmin(distances, axis=1)
    matching_distances = np.min(distances, axis=1)

    matches = select_matches(patches1, patches2, corr_indices)

def select_matches(patches1, patches2, corr_indices):
    corr_points = []

    for i, patch in enumerate(patches1):
        corr_points.append([patch, patches2[corr_indices[i]]])

    return np.array(corr_points)

def calculate_distances(desc1, desc2):
    from scipy.spatial.distance import cdist
    return cdist(desc1, desc2)

def extract_patches(img, keypoints, width, height):
    patches = []
    border_height = int(height/2)
    border_width = int(width/2)
    padded_image = cv2.copyMakeBorder(img, border_height, border_height, border_width, border_width, cv2.BORDER_REPLICATE)

    for keypoint in keypoints:
        row1 = keypoint[0] - border_height
        row2 = keypoint[0] + border_height + 1
        col1 = keypoint[1] - border_width
        col2 = keypoint[1] + border_width + 1
        patches.append(padded_image[row1:row2, col1:col2].flatten())

    return np.array(patches)

def detect_corners(img, blockSize, ksize, k, thresh):
    dst = cv2.cornerHarris(img, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)

    cimg = dst > thresh*dst.max()
    keypoints = np.column_stack(np.where(cimg))

    return keypoints

def ransac(distances, s, N, d, T):
    #N = np.log(1-p)/np.log(1-(1-e)**s)

    for i in range(N):
        sample = distances[np.random.randint(0, len(distances), s)]

if __name__ == "__main__":
    main()