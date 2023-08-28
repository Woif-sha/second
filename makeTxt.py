import os
import xml.etree.ElementTree as ET


def convert(size, box):
    """
    将边界框的坐标转换为相对于输入图片大小的归一化坐标
    Args:
        size: 输入图片的大小
        box: 边界框的坐标
    Returns:
        (x,y,w,h): (x,y)为中心点坐标，w为宽度，h为高度
    """
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)


def convert_annotation(xml_files_path, save_txt_files_path, classes):
    """
    将xml文件转换为txt文件
    Args:
        xml_files_path:输入xml文件的路径
        save_txt_files_path: 输入txt文件的路径
        classes: 项目的类别(list类型)
    Returns:
        None
    """
    xml_files = os.listdir(xml_files_path)
    for xml_name in xml_files:
        print(xml_name)
        xml_file = os.path.join(xml_files_path, xml_name)  # xml文件的路径
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')  # 生成txt文件的路径
        out_txt_f = open(out_txt_path, 'w')  # 打开txt文件
        # 解析xml文件
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text  # 获取difficult标签
            cls = obj.find('name').text  # 获取类别
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            # 获取边界框的坐标
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)  # 转换坐标
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')  # 写入txt文件


if __name__ == "__main__":
    classes1 = ['red', 'blue', 'black']  # 需要转换的类别，需要一一对应
    xml_files1 = 'trainLable/vocLable'  # voc格式的xml标签文件路径
    save_txt_files1 = 'trainLable/txtLable'  # 转化为yolo格式的txt标签文件存储路径
    convert_annotation(xml_files1, save_txt_files1, classes1)  # 转换标签
