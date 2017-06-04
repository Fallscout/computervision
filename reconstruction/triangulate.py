import numpy as np
import cv2

def triangulate(camera1, camera2, matches):
    points1 = matches[:, :2].T
    points2 = matches[:, 2:].T
    result = cv2.triangulatePoints(camera1, camera2, points1, points2).T
    result /= result[:, 3].reshape(-1, 1)
    return result[:, :3]

def fit_fundamental(matches):
    """
    # Own implementation does not work correctly

    A = np.array([[x*x_prime, x*y_prime, x, x_prime*y, y*y_prime, y, x_prime, y_prime, 1] for x, y, x_prime, y_prime in matches])
    u, d, v = np.linalg.svd(A)
    F = v[:, v.shape[1]-1].reshape(3, 3)
    u, d, v = np.linalg.svd(F)
    d[2] = 0
    F = u.dot(np.diag(d)).dot(v)
    F /= F[2, 2]
    return F
    """

    base_points = np.array([x[:2] for x in matches])
    target_points = np.array([x[2:] for x in matches])

    F, mask = cv2.findFundamentalMat(base_points, target_points)

    return F

def visualize_matches(img1, img2, matches):
    import matplotlib.pyplot as plt
    img1_rgb = img1[...::-1]
    img2_rgb = img2[...::-1]
    fig = plt.figure()
    #...

def load_data(img1_path, img2_path, matches_path, camera1_path, camera2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    matches = np.loadtxt(matches_path)
    camera1 = np.loadtxt(camera1_path)
    camera2 = np.loadtxt(camera2_path)
    return img1, img2, matches, camera1, camera2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="")
    parser.add_argument("--img1", default="house1.jpg", type=str, help="")
    parser.add_argument("--img2", default="house2.jpg", type=str, help="")
    parser.add_argument("--matches", default="house_matches.txt", type=str, help="")
    parser.add_argument("--camera1", default="house1_camera.txt", type=str, help="")
    parser.add_argument("--camera2", default="house2_camera.txt", type=str, help="")
    args = parser.parse_args()

    img1, img2, matches, camera1, camera2 = load_data(args.img1, args.img2, args.matches, args.camera1, args.camera2)
    F = fit_fundamental(matches)
    print("Fundamental matrix:\n{}".format(F))

    points1 = np.concatenate((matches[:, :2], np.ones((matches.shape[0], 1))), axis=1)
    points2 = np.concatenate((matches[:, 2:], np.ones((matches.shape[0], 1))), axis=1)
    L = np.dot(F, points1.T)
    errors = np.diag(points2.dot(L))
    residual = sum(errors)
    print(residual)

    world_points = triangulate(camera1, camera2, matches)