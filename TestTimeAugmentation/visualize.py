import glob
from generate_xml_cvat import get_labels_from_json
import os
import cv2
import xml.etree.ElementTree as ET


def visualize_images(pathImg):
    """
    path where images and corresponsing xml files are.
    """
    labels,data = get_labels_from_json("/mnt/data/dataorig/annotations/instances_default.json")
    _images = glob.glob("/mnt/data/dataorig/images/*.jpg")
    _images.extend(glob.glob("/mnt/data/dataorig/images/*.png"))
    for image in data['images']:
        tree = ET.parse(os.path.join(pathImg+"output", os.path.basename(image['file_name']))[:-4]+'.xml')
        root = tree.getroot()   
        boxes = []
        for obj in root.findall('object'):
            boxes.append([obj.find('name').text,float(list(obj.iter('xmin'))[0].text),float(list(obj.iter('ymin'))[0].text),float(list(obj.iter('xmax'))[0].text),float(list(obj.iter('ymax'))[0].text)])
        print("image", image, "boxes: ", len(boxes))
        draw_boxes(os.path.basename(image['file_name']), boxes)


def draw_boxes(img_path,boxes):

    frame = cv2.imread("/mnt/data/dataorig/images/"+img_path,cv2.IMREAD_UNCHANGED)
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for box in boxes:
        cv2.rectangle(frame, (int(box[1]),int(box[2])), (int(box[3]),int(box[4])), (255,0,0) , 5)
    print("storing output :", os.path.join("/mnt/output/", img_path))
    cv2.imwrite(os.path.join("/mnt/output/", img_path), frame)
