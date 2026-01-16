import cv2
import numpy as np
import os
import argparse
import glob
import json

def undistort_images(originalimg, labelpaths, transformedpath):
    for labelpath in labelpaths:
        with open(labelpath, 'r') as file:
            data = json.load(file)

        label = labelpath.split("\\")[-1].split(".json")[0]
        img = cv2.imread(os.path.join(originalimg, label + '.png'))
        IMGWIDTH = img.shape[1]
        IMGHEIGHT = img.shape[0]

        dst_pts = np.array([
            [200, 50],
            [870, 50],
            [870, 700],
            [200, 700]
        ], dtype="float32")

        all_coordinates = []
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'point':
                for point in shape.get('points', []):
                    all_coordinates.append(tuple(point))

        # Convert all_coordinates to a numpy array
        all_coordinates = np.array(all_coordinates)

        # Find the four corner points
        s = all_coordinates.sum(axis=1)          # x + y
        d = all_coordinates[:, 0] - all_coordinates[:, 1]  # x - y

        left_top     = all_coordinates[np.argmin(s)]
        right_bottom = all_coordinates[np.argmax(s)]
        left_bottom  = all_coordinates[np.argmin(d)]
        right_top    = all_coordinates[np.argmax(d)]

        side1 = np.linalg.norm(left_top - right_top)      
        side2 = np.linalg.norm(left_top - left_bottom)

        corner_points = np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(corner_points, dst_pts)

        transformed = cv2.warpPerspective(img, M, (IMGWIDTH, IMGHEIGHT))

        output_filepath = os.path.join(transformedpath, f"{label}.jpg")
        cv2.imwrite(output_filepath, transformed)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Correct system error of the ear image")

    # Add arguments
    parser.add_argument('-i', '--image_folder', default='./images/', type=str, required=False, help='Path to the original image folder')
    parser.add_argument('-l', '--label_folder', default='./labels/', type=str, required=False, help='Path to the label folder')
    parser.add_argument('-o', '--output_undistorted_image_path', default='./transformed/', type=str, required=False, help='Output undistorted image path')

    # Parse the arguments
    args = parser.parse_args()

    original_image_folder = args.image_folder
    label_folder = glob.glob(os.path.join(args.label_folder, '*.json'))
    output_path = args.output_undistorted_image_path


    undistort_images(original_image_folder, label_folder, output_path)
