import os
import subprocess
from multiprocessing import Pool
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import shutil
import cv2 as cv
from shapely.geometry import Polygon

def run_command(command, timeout=120):
    """运行命令，并在超时时间内完成"""
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Command {command} timed out after {timeout} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\nExit code: {e.returncode}\nError message: {e}")

def delete_file(files):
    files_to_delete = glob.glob(files)
    # 删除匹配的文件
    for file in files_to_delete:
        os.remove(file)

# 定义函数
def polygon_to_bbox(polygon):
    # Convert polygon to bounding box
    x_coords, y_coords = zip(*polygon)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    w = round(max_x - min_x, 7)
    h = round(max_y - min_y, 7)

    return min_y, max_y, h, min_x, max_x, w

def extract_ear_rotation_marks(image_file):
    # 转换为HSV格式方便色块分割
    image_hsv = cv.imread(image_file)
    image_hsv = cv.cvtColor(image_hsv[-800:, :, :3], cv.COLOR_BGR2HSV)

    # 最小轮廓面积阈值，用于去除小面积噪音轮廓
    MIN_CONTOUR_AREA = 100  # 根据需要调整该值

    ear_rotation_marks = []  # 初始化每个类的多边列表

    # 设置色调值
    hsv_colors = [np.array([90, 245, 135])]

    for hsv_color in hsv_colors:
        # 定义颜色范围
        lower_bound = np.array(hsv_color) - np.array([10, 10, 120])
        upper_bound = np.array(hsv_color) + np.array([10, 10, 120])

        # 创建蒙版
        mask = cv.inRange(image_hsv, lower_bound, upper_bound)

        # 查找颜色区域的轮廓
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 根据面积过滤小轮廓
            area = cv.contourArea(contour)
            if area >= MIN_CONTOUR_AREA:  # 保留大于最小面积的轮廓
                polygon = Polygon(contour[:, 0, :])  # 创建多边形
                ear_rotation_marks.append(polygon.centroid.x)  # 获取叶节点的高度---这里的坐标原点在左下

    # Extract the head and tail elements
    ear_rotation_marks.sort()
    if len(ear_rotation_marks) >= 7:
        marker_dis = int(ear_rotation_marks[6]) - int(ear_rotation_marks[0])
        if marker_dis < 300 or marker_dis > 500:
            return 420
        else:
            return marker_dis
    else:
        return 420

def process_file(full_run_with_extension, pic_path, model_path, image_undistort_parameters, ear_center):
    # Remove file extension
    run = os.path.basename(full_run_with_extension).split('.')[0]
    full_run = os.path.normpath(os.path.join(os.path.dirname(full_run_with_extension), run))

    ears_dir = os.path.join(pic_path, 'ear')
    projections_dir = os.path.join(pic_path, 'projection')

    # Import video processing functions
    import sys
    sys.path.append(model_path)
    from camera_functions import undistort_and_resize_image

    mtx, dist = image_undistort_parameters[0], image_undistort_parameters[1]

    # Extract frames from video
    run_command(
        ["ffmpeg", '-i', full_run_with_extension, f'{full_run}_frame%03d.png']
    )

    # Remove image distortion
    images = glob.glob(f"{full_run}_frame*.png")

    for fname in images:
        undistort_and_resize_image(fname, mtx, dist, 1072, 1440)

    delete_file(f"{full_run}_frame*.png")

    ear_center = int(ear_center * 1072)

    # Stitch images
    run_command(
        ["convert", f"{full_run}_undistort*.png", "-crop", f"1x1440+{ear_center}+0", "+repage", f"{full_run}_pixel%03d.png"]
    )

    # Copy specific images to the ears directory
    for i in [100, 124, 148, 172, 196, 220]:
        src_file = f"{full_run}_undistort{i}.png"
        dest_file = os.path.join(ears_dir, f"{run}_{(i // 24) - 3}.png")
        shutil.move(src_file, dest_file)
    delete_file(f"{full_run}_undistort*.png")

    # Combine pixel images horizontally
    run_command([
        "convert", f"{full_run}_pixel*.png", "+append", "+repage", f"{full_run}_raw.png"
    ])
    delete_file(f"{full_run}_pixel*.png")

    marker_dis = extract_ear_rotation_marks(f"{full_run}_raw.png")
    resize_length = int(marker_dis * 1.1)

    # Crop raw image
    run_command([
        "convert", f"{full_run}_raw.png", "-crop", f"{resize_length}x1440+40+0", "+repage", f"{full_run}_cropped.png"
    ])
    delete_file(f"{full_run}_raw.png")

    # Final resize operation
    run_command([
        "convert", f"{full_run}_cropped.png", "-resize", "1920x1440!", f"{full_run}.png"
    ])
    delete_file(f"{full_run}_cropped.png")

    # Move the generated image
    shutil.move(f'{full_run}.png', projections_dir)


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Process videos into projections")

    # Add arguments
    parser.add_argument('-v', '--video_folder', default='./videos/', type=str, required=False, help='Path to the original video folder')
    parser.add_argument('-p', '--parameter_folder', default='./image_process/', type=str, required=False,
                        help='Path to the image undistortion parameter folder')
    parser.add_argument('-o', '--output_path',  default='./composition/', type=str, required=False, help='Output image path')
    parser.add_argument('-c', '--cores_number', default=5, type=int, required=False,
                        help='Number of cores used for parallel processing')
    parser.add_argument('-e', '--ear_center', default=0.5, type=float, required=False,
                        help='Ear center in the image used for video processing')


    # Parse the arguments
    args = parser.parse_args()

    # Create necessary directories under the output path
    ear_dir = os.path.join(args.output_path, 'ear/')
    projection_dir = os.path.join(args.output_path, 'projection/')
    
    os.makedirs(ear_dir, exist_ok=True)
    os.makedirs(projection_dir, exist_ok=True)

    # 导入视频处理参数
    mtx = np.load(os.path.join(args.parameter_folder, "HQ_camera_1072_1440_mtx_dist.npz"))["x"]
    dist = np.load(os.path.join(args.parameter_folder, "HQ_camera_1072_1440_mtx_dist.npz"))["y"]
    image_undistort_parameters = [mtx, dist]

    vid_path = os.path.abspath(args.video_folder)
    all_files = [os.path.join(vid_path, f) for f in os.listdir(vid_path) if f.endswith(('.mp4', '.avi'))]
    tasks = [(file, args.output_path, args.parameter_folder, image_undistort_parameters, args.ear_center) for file in all_files]

    with ThreadPoolExecutor(max_workers=args.cores_number) as executor:
        executor.map(lambda p: process_file(*p), tasks)
