import glob
from generate_xml_cvat import get_labels_from_json
import os
import cv2

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
        for obj in root.findall('object'):
            draw_boxes(os.path.basename(image['file_name']), obj.find('name').text,float(list(obj.iter('xmin'))[0].text),float(list(obj.iter('ymin'))[0].text),float(list(obj.iter('xmax'))[0].text),float(list(obj.iter('ymax'))[0].text))


def draw_boxes(img_path,label, xmin, ymin, xmax, ymax):
    frame = cv2.imread("/mnt/data/dataorig/images"+img_path)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0) , 5)
  