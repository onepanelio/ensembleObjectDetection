import testTimeAugmentation
import function
import os
import shutil
import argparse
import ensembleOptions
from mainModel import models
from imutils import paths
import argparse
import warnings
warnings.filterwarnings('ignore')



def main(args):
    listModels = []
    models_list = args.models.split(",")
    print("Models to be run: ", models_list)
    if 'mask_rcnn' in models_list:
        maskRcnn = testTimeAugmentation.MaskRCNNPred('/mnt/src/mask_rcnn_coco.h5', '/mnt/src/coco.names')
        listModels.append(maskRcnn)
    if 'retinanet' in models_list:
        retinaResnet50 = testTimeAugmentation.RetinaNetResnet50Pred('/mnt/src/resnet50_coco_best_v2.1.0.h5', '/mnt/src/coco.csv')
        listModels.append(retinaResnet50)
    if 'yolo_darknet' in models_list:
        yoloDarknet = testTimeAugmentation.DarknetYoloPred('/mnt/src/yolov3.weights', '/mnt/src/coco.names','/mnt/src/yolov3.cfg')
        listModels.append(yoloDarknet)
    if 'ssd_resnet' in models_list:
        ssdResnet = testTimeAugmentation.MXnetSSD512Pred('/mnt/src/ssd_512_resnet50_v1_voc-9c8b225a.params', '/mnt/src/classesMXnet.txt')
        listModels.append(ssdResnet)
    if 'faster_resnet' in models_list:
        fasterResnet = testTimeAugmentation.MXnetFasterRCNNPred('/mnt/src/faster_rcnn_resnet50_v1b_voc-447328d8.params', '/mnt/src/classesMXnet.txt')
        listModels.append(fasterResnet)
        
#     listaModels = [retinaResnet50, maskRcnn]
    models(listModels,args.images_path,args.option, args.combine)
    print(os.listdir('/mnt/src/'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default="retinanet,yolo_darknet")
    parser.add_argument('--images_path', default='/data/images')
    parser.add_argument('--option', default='unanimous')
    parser.add_argument("--combine", default=False)
    args = parser.parse_args()
    main(args)
