import os
import random
import re
import shutil
from pathlib import Path

import cv2
import torch

from models.experimental import attempt_load
from utils.datasetsCartoon import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

imgsz = 640  # 设置输入图片的大小
conf_thres = 0.25  # 设置置信度阈值
iou_thres = 0.45  # 设置iou阈值
result_list = []  # 保存检测结果到列表中


def clean_dir(folder_path):
    """
    清空文件夹内容
    Args:
        folder_path: 输入文件夹的路径
    Returns:
        None
    """
    if os.path.exists(folder_path):  # 检查文件夹是否存在
        shutil.rmtree(folder_path)  # 清空文件夹内容

    os.mkdir(folder_path)  # 创建文件夹


def detect(path, weights, save_dir, show):
    """
    检测人群并输出结果
    Args:
        path:输入文件夹或文件的路径
        weights:选择使用的模型
        save_dir:保存结果的文件夹
        show:是否显示图片
    Returns:
        None
    """
    global imgsz, conf_thres, iou_thres

    # 清空保存结果的文件夹
    clean_dir(save_dir)

    # Directories
    save_dir = Path(save_dir)  # 保存结果的文件夹

    # Initialize
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # print("imgsz:",imgsz)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[0, 0, 255], [255, 0, 0], [0, 0, 0]]  # red,blue,black

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    for path, img, im0s, filename in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # expand for batch dim

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print(img.shape)
                print(im0.shape)
                num = 0
                ans = {'红色': 0, '蓝色': 0, '黑色': 0}
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if int(c) == 0:
                        ans['红色'] = n
                    elif int(c) == 1:
                        ans['蓝色'] = n
                    else:
                        ans['黑色'] = n
                    num += n  # 人数求和
                s = f"红色{ans.get('红色')}人，蓝色{ans.get('蓝色')}人，黑色{ans.get('黑色')}人"  # 人群统计结果

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            result = f'图片“{filename}”中人群总数{num}人；{s}。'
            print(result)  # 在终端中输出结果
            result_list.append(result)  # 将结果信息保存到列表中，便于排序
            cv2.imwrite(save_path, im0)  # 保存结果图片
            # 显示图片
            if show:
                im0 = cv2.resize(im0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # 缩放图片到原来的一半
                cv2.imshow(f'{filename}', im0)


def write_file(result_list, save_dir):
    """
    将检测结果写入文件
    Args:
        result_list: 检测结果列表
        save_dir: 保存结果的文件夹
    Returns:
        None
    """
    save_dir = Path(save_dir)

    def sort_key(x):
        """
        按照文件名的序号排序
        Args:
            x: 文件名
        Returns:
            None
        """
        match = re.search(r"TZ\d+_No(\d+)\.jpg", x)  # 匹配图片编号
        if match:
            image_number = int(match.group(1))
            return image_number

    result_list = sorted(result_list, key=sort_key)
    file_path = Path(save_dir / 'result.txt')  # 构建保存路径

    # 将排序后的结果写入新文件
    with open(file_path, "w+") as file:
        for item in result_list:
            file.write(item + '\n')


def sample(fileDir="images" + "//", tarDir="detectImg" + "//"):
    """
    随机抽取二十张图片
    Args:
        fileDir: 源图片文件夹路径
        tarDir: 抽样图片的文件夹路径
    Returns:
        None
    """

    def moveFile(fileDir, tarDir):
        """
        随机复制二十个文件
        Args:
            fileDir: 源图片文件夹路径
            tarDir:  抽样图片的文件夹路径
        Returns:
            None
        """
        pathDir = os.listdir(fileDir)  # 取图片的原始路径
        number = 20
        sample = random.sample(pathDir, number)  # 随机选取20张样本图片
        for name in sample:
            shutil.copy(fileDir + name, tarDir + name)  # 复制到抽样文件夹
        return

    if os.path.exists(tarDir):
        shutil.rmtree(tarDir)  # 清空抽样文件夹里面的全部内容 (保持文件数量不变)
    os.mkdir(tarDir)  # 重新创建抽样文件夹
    moveFile(fileDir, tarDir)  # 进行随机抽样


def main(path='images', weights='weights/yolov5s_cartoon.pt', save_dir='result', show=False):
    """
    主函数
    Args:
        path:输入文件夹路径
        weights:选择使用的模型权重
        save_dir:保存结果的文件夹
        show:是否显示图片
    Returns:
        None
    """
    # sample(fileDir=path + "//", tarDir="detectImg" + "//")  # 随机抽取二十张图片并将结果保存到detectImg文件夹中
    with torch.no_grad():
        detect(path=path, weights=weights, save_dir=save_dir, show=show)  # 进行人群检测
    write_file(result_list, save_dir)  # 将结果写入文件
    # 防止图片一闪而过
    if show:
        cv2.waitKey()


if __name__ == '__main__':
    main(path='trainData/trainImages/P3_No20.jpg', weights='weights/yolov5s_cartoon.pt', save_dir='result', show=False)
