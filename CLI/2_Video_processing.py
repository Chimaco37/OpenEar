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
from PIL import Image
from pathlib import Path

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
    image_hsv = cv.cvtColor(image_hsv[-300:, :, :3], cv.COLOR_BGR2HSV)

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
        print(marker_dis)
        if marker_dis < 300 or marker_dis > 500:
            return 420
        else:
            return marker_dis
    else:
        return 420

# Crop the frames
def crop_frames(frame_files, output_pattern):
    for idx, frame_path in enumerate(frame_files):
        with Image.open(frame_path) as img:
            width, height = img.size
            # 裁剪区域 (left, upper, right, lower)
            box = (width/2, 0, width/2+1, height)
            cropped = img.crop(box)
            cropped.save(output_pattern % idx)

def combine_images_horizontally(file_paths, output_path):
    """水平拼接匹配到的所有图像"""
    images = []
    try:
        images = [Image.open(fp) for fp in file_paths]
        # 创建画布
        combined = Image.new('RGB', (len(file_paths), 1440))
        # 拼接图像
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += 1
        # 保存结果
        combined.save(output_path, quality=95, optimize=True)
    finally:
        # 确保关闭所有图像
        for img in images:
            img.close()

def process_file(full_run_with_extension, pic_path, model_path, image_undistort_parameters):
    # Remove file extension
    full_path = Path(full_run_with_extension)
    run = full_path.stem
    full_run = full_path.parent / run

    # 调用工具时，使用这个路径
    FFMPEG = Path(model_path) / "ffmpeg.exe"

    ears_dir = os.path.join(pic_path, 'ear')
    projections_dir = os.path.join(pic_path, 'projection')

    # Import video processing functions
    import sys
    sys.path.append(model_path)
    from image_process.camera_functions import undistort_and_resize_image

    mtx, dist = image_undistort_parameters[0], image_undistort_parameters[1]

    # Extract frames from video
    run_command(
        [FFMPEG, '-i', full_run_with_extension, f'{full_run}_frame%03d.png']
    )

    # Remove image distortion
    images = glob.glob(f"{full_run}_frame*.png")

    for fname in images:
        undistort_and_resize_image(fname, mtx, dist, 1072, 1440)

    delete_file(f"{full_run}_frame*.png")

    # Crop the frame to preset height
    pattern = f"{run}_undistort*.png"
    frame_files = sorted(full_path.parent.glob(pattern))

    crop_frames(
        frame_files=frame_files,
        output_pattern=f"{full_run}_pixel%03d.png",
    )

    # Copy specific images to the ears directory
    for i in [100, 116, 132, 148, 164, 180]:
        src_file = f"{full_run}_undistort{i}.png"
        dest_file = os.path.join(ears_dir, f"{run}_{(i // 16) - 5}.png")
        shutil.move(src_file, dest_file)
    delete_file(f"{full_run}_undistort*.png")

    # Combine pixel images horizontally
    pattern = f"{run}_pixel*.png"
    file_paths = sorted(full_path.parent.glob(pattern))

    # 使用示例
    combine_images_horizontally(
        file_paths=file_paths,
        output_path=f"{full_run}_raw.png"
    )
    delete_file(f"{full_run}_pixel*.png")

    marker_dis = extract_ear_rotation_marks(f"{full_run}_raw.png")
    resize_length = int(marker_dis * 1.1)

    # Crop raw image
    with Image.open(f"{full_run}_raw.png") as img:
        width, height = img.size
        # 执行裁剪 (left, upper, right, lower)
        box = (40, 0, 40+resize_length, height)
        cropped = img.crop(box)

        # 直接保存最终 raw 文件（跳过临时文件）
        cropped.save(f"{full_run}_cropped.png", "PNG")
    delete_file(f"{full_run}_raw.png")

    # Final resize operation
    target_size = (1920, 1440)
    with Image.open(f"{full_run}_cropped.png") as img:
        # 移除 Alpha 通道（转换为 RGB）
        rgb_img = img.convert("RGB")

        # 强制拉伸到目标尺寸（LANCZOS 重采样保持质量）
        resized = rgb_img.resize(target_size, Image.Resampling.LANCZOS)

        # 保存最终结果
        resized.save(f"{full_run}.png", "PNG", optimize=True, quality=95)
    delete_file(f"{full_run}_cropped.png")

    # Move the generated image
    shutil.move(f'{full_run}.png', projections_dir)


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Process videos into projections")

    # Add arguments
    parser.add_argument('-v', '--video_folder', default='./videos/', type=str, required=False, help='Path to the original video folder')
    parser.add_argument('-m', '--model_folder', default='./models/', type=str, required=False,
                        help='Path to the model folder')
    parser.add_argument('-o', '--output_path',  default='./composition/', type=str, required=False, help='Output image path')
    parser.add_argument('-c', '--cores_number', default=5, type=int, required=False,
                        help='Number of cores used for parallel processing')


    # Parse the arguments
    args = parser.parse_args()

    # Create necessary directories under the output path
    ear_dir = os.path.join(args.output_path, 'ear/')
    projection_dir = os.path.join(args.output_path, 'projection/')
    
    os.makedirs(ear_dir, exist_ok=True)
    os.makedirs(projection_dir, exist_ok=True)

    # 导入视频处理参数
    mtx = np.load(os.path.join(args.model_folder, "image_process", "HQ_camera_1072_1440_mtx_dist.npz"))["x"]
    dist = np.load(os.path.join(args.model_folder, "image_process", "HQ_camera_1072_1440_mtx_dist.npz"))["y"]
    image_undistort_parameters = [mtx, dist]

    vid_path = os.path.abspath(args.video_folder)
    all_files = [os.path.join(vid_path, f) for f in os.listdir(vid_path) if f.endswith(('.mp4', '.avi'))]
    tasks = [(file, args.output_path, args.model_folder, image_undistort_parameters) for file in all_files]

    with ThreadPoolExecutor(max_workers=args.cores_number) as executor:
        executor.map(lambda p: process_file(*p), tasks)
