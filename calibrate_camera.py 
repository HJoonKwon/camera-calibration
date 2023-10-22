import os, json, glob, math
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import pprint 

def evaluate(points_2d_list, points_3d_list, K, dist, rvecs, tvecs):
    mean_error = 0
    for i in range(len(points_3d_list)):
        imgpoints2, _ = cv2.projectPoints(points_3d_list[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(points_2d_list[i], imgpoints2, cv2.NORM_L2)
        mean_error += error**2
    total = len(points_3d_list) * points_3d_list[0].shape[0]
    return math.sqrt(mean_error/total)

def calibrate(img_dir: str, output_dir: str, dims=(8, 5), visualize=False):
    assert os.path.exists(img_dir), "Image directory does not exist"
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(img_dir + "/*.jpg")
    row, col = dims

    points_3d = np.zeros((row * col, 3), np.float32)
    points_3d[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    # (40, 3)
    # [[0, 0, 0], [0, 1, 0], [0, 2, 0], ... ,[7, 4, 0]]

    points_2d_list = []
    points_3d_list = []

    for img_path in tqdm(images):
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (row, col), None)
        if ret:
            corners = cv2.cornerSubPix(
                gray_img,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            if visualize:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (row, col), corners, ret)
                cv2.drawChessboardCorners(img, (row, col), corners, ret)
                cv2.imshow("img", img)
                cv2.waitKey(200)

            # 2d corner points and corresponding 3d points
            points_2d_list.append(corners)
            points_3d_list.append(points_3d)

    ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_3d_list, points_2d_list, gray_img.shape[::-1], None, None
    )
    
    print(f"Evaluate...")
    mean_error = evaluate(points_2d_list, points_3d_list, mat, dist, rvecs, tvecs)
    assert abs(mean_error - ret) < 1e-6, "Mean error does not match"
    print( "Mean reprojection error: {}".format(mean_error) )

    rmats = []
    for rvec in rvecs:
        rmat = cv2.Rodrigues(rvec)[0]
        rmats.append(rmat.tolist())
    
    formatted_mat = pprint.pformat(mat, indent=4)
    formatted_dist = pprint.pformat(dist, indent=4)
    formatted_rvec = pprint.pformat(rvecs[0], indent=4)
    formatted_tvec = pprint.pformat(tvecs[0], indent=4)
    formatted_rmat = pprint.pformat(rmats[0], indent=4)
    print(f"Camera Matrix:\n{formatted_mat}")
    print(f"Distortion Coefficients:\n{formatted_dist}")
    print(f"0th Rotation Vector:\n{formatted_rvec}")
    print(f"0th Translation Vector:\n{formatted_tvec}")
    print(f"0th Rotation Matrix:\n{formatted_rmat}")
    
    output = {
        'K': mat.tolist(),
        'distortion': dist.tolist(),
        'rvecs': [rvec.tolist() for rvec in rvecs],
        'tvecs': [tvec.tolist() for tvec in tvecs],
        'rmats': rmats,
    }
    return output 

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--img_dir", type=str, default="./images")
    argparse.add_argument("--output_dir", type=str, default="./output")
    argparse.add_argument("--dims", type=tuple, default=(8, 5))
    argparse.add_argument("--visualize", action="store_true")
    args = argparse.parse_args()
    output = calibrate(**vars(args))
    with open(os.path.join(args.output_dir, 'output.json'), 'w') as f:
        json.dump(output, f)