import numpy as np
import cv2

def triangulate(matchpoints1, matchpoints2, camera1, camera2):
    result = cv2.triangulatePoints(camera1, camera2, matchpoints1.T, matchpoints2.T).T
    result /= result[:, 3].reshape(-1, 1)
    return result[:, :3]

def FMatrix(matchepoints1, matchpoints2):
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
    return cv2.findFundamentalMat(matchepoints1, matchpoints2)[0]

def visualize_matches(points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="black", s=0.5)
    fig.show()

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
    
    matchpoints1 = np.array([x[:2] for x in matches])
    matchpoints2 = np.array([x[2:] for x in matches])

    F = FMatrix(matchpoints1, matchpoints2)
    print("Fundamental matrix:\n{}".format(F))

    points1 = np.concatenate((matches[:, :2], np.ones((matches.shape[0], 1))), axis=1)
    points2 = np.concatenate((matches[:, 2:], np.ones((matches.shape[0], 1))), axis=1)
    L = np.dot(F, points1.T)
    errors = np.diag(points2.dot(L))
    residual = sum(errors)
    print(residual)

    world_points = triangulate(matchpoints1, matchpoints2, camera1, camera2)
    visualize_matches(world_points)