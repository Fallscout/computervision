import numpy as np
import cv2

def im_stitch(img_left, img_right, naive=False, patchsize=3, xcorr=False):
    if patchsize % 2 != 1:
        raise ValueError("patchsize must be odd.")

    if naive:
        # doesn't work properly
        gray_left = np.float32(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY))
        gray_right = np.float32(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))

        kps_left = detect_corners(gray_left, 3000, 0.01, 10)
        kps_right = detect_corners(gray_right, 3000, 0.01, 10)

        features_left = extract_patches(gray_left, kps_left, patchsize)
        features_right = extract_patches(gray_right, kps_right, patchsize)
    else:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        (kps_left, features_left) = sift.detectAndCompute(gray_left, None)
        (kps_right, features_right) = sift.detectAndCompute(gray_right, None)

        kps_left = np.float32([kp.pt for kp in kps_left])
        kps_right = np.float32([kp.pt for kp in kps_right])

    print("Number of keypoints in left image: {}".format(len(kps_left)))
    print("Left features shape: {}".format(features_left.shape))
    print("Number of keypoints in right image: {}".format(len(kps_right)))
    print("Right features shape: {}".format(features_right.shape))

    if xcorr:
        # doesn't work properly
        distances = calculate_distances(features_right, features_left, "xcorr")
        corr_indices = np.argmax(distances, axis=1)
    else:
        distances = calculate_distances(features_right, features_left)
        corr_indices = np.argmin(distances, axis=1)

    matching_distances = np.min(distances, axis=1)
    matching_keypoints = select_matches(kps_right, kps_left, corr_indices)
    h = ransac(matching_keypoints, 4, 300, 5, 500)

    print("Homography:\n{}".format(h))

    result = cv2.warpPerspective(img_right, h, (img_left.shape[1] + img_right.shape[1], max(img_left.shape[0], img_right.shape[0])))
    result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

    return result

def select_matches(patches1, patches2, corr_indices):
    corr_points = []

    for i, patch in enumerate(patches1):
        corr_points.append([patch, patches2[corr_indices[i]]])

    return np.array(corr_points)

def calculate_distances(desc1, desc2, metric="euclidean"):
    if metric == "euclidean":
        from scipy.spatial.distance import cdist
        return cdist(desc1, desc2)
    elif metric == "xcorr":
        result = []
        for d1 in desc1:
            row = []
            for d2 in desc2:
                row.append(np.correlate(d1, d2).item())
            result.append(row)
        return np.array(result)

def extract_patches(img, keypoints, patchsize):
    patches = []
    border_height = int(patchsize/2)
    border_width = int(patchsize/2)
    padded_image = cv2.copyMakeBorder(img, border_height, border_height, border_width, border_width, cv2.BORDER_REPLICATE)

    for keypoint in keypoints:
        row1 = int(keypoint[0] - border_height)
        row2 = int(keypoint[0] + border_height + 1)
        col1 = int(keypoint[1] - border_width)
        col2 = int(keypoint[1] + border_width + 1)
        patches.append(padded_image[row1+border_height:row2+border_height, col1+border_width:col2+border_width].flatten())

    return np.array(patches)

def detect_corners(img, maxCorners, qualityLevel, minDistance):
    result = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
    # Need to swap columns
    return result.reshape(result.shape[0], 2)[:, [1, 0]]

def ransac(matches, s, N, d, T):
    """
    # Own implementation does not work correctly

    base_points = np.concatenate([np.array([match[0] for match in matches]), np.ones((len(matches), 1))], axis=1)
    target_points = np.concatenate([np.array([match[1] for match in matches]), np.ones((len(matches), 1))], axis=1)
    points = np.array([point.flatten() for point in matches])
    num_inliers = []
    models = []

    for i in range(N):
        indices = np.random.randint(0, matches.shape[0], s)
        model_points = points[indices]
        A1 = np.array([[-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime] for x, y, x_prime, y_prime in model_points])
        A2 = np.array([[0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime] for x, y, x_prime, y_prime in model_points])
        A = np.concatenate([A1, A2], axis=0)

        u, _, v = np.linalg.svd(A)

        last_col = v[:, v.shape[1]-1]
        last_col /= last_col[last_col.size-1]
        maybe_model = last_col.reshape(3, 3)
        result = np.dot(base_points, maybe_model)
        #result /= result[:, 2].reshape(-1, 1)
        distances = np.linalg.norm(result - target_points, axis=1)
        inliers = np.where(distances < d)[0]

        if inliers.size >= T:
            return maybe_model
        
        num_inliers.append(inliers.size)
        models.append(maybe_model)

    return models[np.argmax(num_inliers)]
    """

    base_points = np.array([match[0] for match in matches])
    target_points = np.array([match[1] for match in matches])
    return cv2.findHomography(base_points, target_points, cv2.RANSAC)[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="")
    parser.add_argument("--left", default="img1.jpg", type=str, help="")
    parser.add_argument("--right", default="img2.jpg", type=str, help="")
    parser.add_argument("--naive", action="store_true", help="")
    parser.add_argument("--xcorr", action="store_true", help="")
    args = parser.parse_args()
    
    img_left = cv2.imread(args.left)
    img_right = cv2.imread(args.right)

    result = im_stitch(img_left, img_right, naive=args.naive, xcorr=args.xcorr)