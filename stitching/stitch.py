import numpy as np
import cv2

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="")
    parser.add_argument("--left", default="img1.jpg", type=str, help="")
    parser.add_argument("--right", default="img2.jpg", type=str, help="")
    parser.add_argument("--use_sift", action="store_true")
    args = parser.parse_args()
    
    img_left = cv2.imread(args.left)
    img_right = cv2.imread(args.right)

    result = im_stitch(img_left, img_right, args.use_sift)

    return result

def im_stitch(img_left, img_right, use_sift=False):

    if use_sift:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        (kps_left, features_left) = sift.detectAndCompute(gray_left, None)
        (kps_right, features_right) = sift.detectAndCompute(gray_right, None)

        kps_left = np.float32([kp.pt for kp in kps_left])
        kps_right = np.float32([kp.pt for kp in kps_right])
    else:
        gray_left = np.float32(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY))
        gray_right = np.float32(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))

        kps_left = detect_corners(gray_left, 2, 3, 0.04, 0.01)
        kps_right = detect_corners(gray_right, 2, 3, 0.04, 0.01)

        features_left = extract_patches(gray_left, kps_left, 3, 3)
        features_right = extract_patches(gray_right, kps_right, 3, 3)
    
    print("Number of keypoints in left image: {}".format(len(kps_left)))
    print("Left features shape: {}".format(features_left.shape))
    print("Number of keypoints in right image: {}".format(len(kps_right)))
    print("Right features shape: {}".format(features_right.shape))

    distances = calculate_distances(features_right, features_left)

    corr_indices = np.argmin(distances, axis=1)
    matching_distances = np.min(distances, axis=1)
    matching_keypoints = select_matches(kps_right, kps_left, corr_indices)
    h, status = ransac(matching_keypoints)

    print(h)

    result = cv2.warpPerspective(img_right, h, (img_left.shape[1] + img_right.shape[1], max(img_left.shape[0], img_right.shape[0])))
    result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

    return result

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
        patches.append(padded_image[row1+border_height:row2+border_height, col1+border_width:col2+border_width].flatten())

    return np.array(patches)

def detect_corners(img, blockSize, ksize, k, thresh):
    dst = cv2.cornerHarris(img, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)

    cimg = dst > thresh*dst.max()
    keypoints = np.column_stack(np.where(cimg))

    return keypoints

def ransac(matches, s=None, N=None, d=None, T=None):
    from sklearn import linear_model
    base_points = np.array([match[0] for match in matches])
    target_points = np.array([match[1] for match in matches])

    h, status = cv2.findHomography(base_points, target_points, cv2.RANSAC)

    return h, status

if __name__ == "__main__":
    result = main()