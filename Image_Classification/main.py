import os
import time
import math
import random
import shutil
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

import models
import numpy as np
from options import parser
from collections import OrderedDict
from utils import *
from data.pacs_dataset import pacs_dataset_read 

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print('Parameters:')
for key, value in state.items():
    print('    {key} : {value}'.format(key=key, value=value))

SINGLE = args.single
T = 3.0

def main(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.get_model(pretrained=args.pretrained, dataset = args.dataset,
                             arch = args.arch, num_classes= args.n_classes, bias=args.bias)
    # pacs
    base_path = './'
    batch_s = 256
    source_dloader, testloader = pacs_dataset_read(base_path,[args.source], args.target, batch_s, target_flg=True)
    text_feature_dim = 512
    text_features_ems = ops.zeros((4, 512), dtype = ms.float32)

    # Evaluate
    if args.evaluate:
        print('Evaluate model')
        top1, top5 = validate(testloader, model,text_features_ems, 0, 
                              (args.lbda, 0), args.den_target)     
        print('Test Acc (Top-1): %.2f, Test Acc (Top-5): %.2f' % (float(top1), float(top5)))
        return

    return

def validate(val_loader, model,text_features_ems, epoch, param, den_target, p=0):
    lbda, gamma = param
    # switch to evaluate mode
    (batch_time, data_time, closses, rlosses, blosses, losses, 
                                            top1, top5, block_flops)= getAvgMeter(9)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (x,targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # get inputs
        batch_size = x.shape[0]
        # inference
        inputs = {"x": x, "label": targets, "den_target": den_target, "lbda": lbda,
              "gamma": gamma, "p": p, "text_features_ems":text_features_ems}
        outputs= model(**inputs)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs["out"], targets, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:}'.format(
            batch=batch_idx+1, size=len(val_loader), bt=batch_time.avg, total=bar.elapsed_td
            )+'| top1: {top1:.2f} | top5: {top5:.2f}'.format(
            top1=float(top1.avg), top5=float(top5.avg))
        bar.next()
    bar.finish()
    return (top1.avg, top5.avg)


def getAvgMeter(num):
    return [AverageMeter() for _ in range(num)]


if __name__ == "__main__":
    if args.task == 'PACS':
        args.Domain_ID = ['art_painting', 'sketch', 'photo', 'cartoon']
        args.classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        args.n_classes = 7
        args.n_domain = 4

    elif args.task == "HOME":
        args.Domain_ID = ['Clipart', 'Art', 'RealWorld', 'Product']
        args.classes = ["Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator",
                        "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp",
                        "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
                        "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker",
                        "Monitor", "Mop", "Mouse", "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil",
                        "Postit_Notes", "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler", "Scissors",
                        "Screwdriver", "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table", "Telephone",
                        "Toothbrush", "Toys", "Trash_Can", "TV", "Webcam"]
        args.n_classes = 65
        args.n_domain = 4
        
    elif args.task == "VLCS":
        args.Domain_ID = ["LABELME", "SUN", "VOC", "CALTECH"]
        args.classes = ["bird", "car", "chair", "dog", "person"]
        args.n_classes = 5
        args.n_domain = 4
        
    elif args.task == "DNET":
        args.Domain_ID = ["clipart", "infograph", "painting", "quickdraw", "real","sketch",]
        args.classes = ['tractor', 'bulldozer', 'tennis_racquet', 'bridge', 'monkey', 'ice_cream', 'castle', 'cactus', 'diamond', 'diving_board', 'broom', 'scissors', 'whale', 'jail', 'parrot', 'nose', 'floor_lamp', 'arm', 'teapot', 'hamburger', 'house', 'bee', 'sun', 'blueberry', 'backpack', 'strawberry', 'oven', 'saw', 'underwear', 'mouth', 'line', 'flip_flops', 'cup', 'shorts', 'vase', 'laptop', 'dog', 'sandwich', 'drums', 'van', 'chair', 'beard', 'cell_phone', 'octagon', 'lipstick', 'stethoscope', 'leaf', 'owl', 'headphones', 'picture_frame', 'hot_tub', 'syringe', 'river', 'raccoon', 'donut', 'foot', 'bowtie', 'microphone', 'piano', 'dumbbell', 'peas', 'baseball_bat', 'hat', 'crab', 'giraffe', 'rake', 'waterslide', 'keyboard', 'grass', 'bush', 'carrot', 'necklace', 'pool', 'rhinoceros', 'pliers', 'frying_pan', 'umbrella', 'cake', 'angel', 'pig', 'pillow', 'submarine', 'campfire', 'violin', 'trumpet', 'animal_migration', 'traffic_light', 'speedboat', 'garden', 'washing_machine', 'squiggle', 'axe', 'computer', 'sleeping_bag', 'hockey_puck', 'wheel', 'swing_set', 'basketball', 'eraser', 'popsicle', 'watermelon', 'elephant', 'streetlight', 'lighthouse', 'bracelet', 'sword', 'light_bulb', 'alarm_clock', 'smiley_face', 'circle', 'moon', 'crayon', 'couch', 'steak', 'purse', 'zebra', 'cello', 'helicopter', 'fire_hydrant', 'anvil', 'fence', 'squirrel', 'spreadsheet', 'barn', 'pizza', 'postcard', 'blackberry', 'crocodile', 'mouse', 'aircraft_carrier', 'eye', 'matches', 'television', 'moustache', 'firetruck', 'teddy-bear', 'rainbow', 'rollerskates', 'panda', 'hot_air_balloon', 'paint_can', 'rifle', 'wristwatch', 'nail', 'pineapple', 'door', 'paintbrush', 'leg', 'apple', 'duck', 'airplane', 'lion', 'fork', 'ambulance', 'feather', 'potato', 'stereo', 'basket', 'dresser', 'truck', 'guitar', 'pear', 'camera', 'chandelier', 'spider', 'broccoli', 'stop_sign', 'palm_tree', 'flashlight', 'toilet', 'candle', 'ant', 'ocean', 'stove', 'lightning', 'bat', 'elbow', 'flamingo', 'remote_control', 'mosquito', 'trombone', 'hand', 'bucket', 'envelope', 'sheep', 'asparagus', 'sailboat', 'jacket', 'triangle', 'skateboard', 'saxophone', 'soccer_ball', 'bear', 'snake', 'garden_hose', 'lollipop', 'cruise_ship', 'golf_club', 'penguin', 'pants', 'calendar', 'sweater', 'rabbit', 'peanut', 'grapes', 'pickup_truck', 'harp', 'stitches', 'paper_clip', 'camel', 'cookie', 'bread', 'beach', 'stairs', 'toe', 'bench', 'passport', 'wine_glass', 'hospital', 'square', 'helmet', 'star', 'drill', 'The_Eiffel_Tower', 'compass', 'hexagon', 'hurricane', 'dolphin', 'binoculars', 'fish', 'power_outlet', 'bandage', 'hammer', 'kangaroo', 'tooth', 'boomerang', 'crown', 'dragon', 'ladder', 'fireplace', 'see_saw', 'bus', 'butterfly', 'tent', 't-shirt', 'ear', 'yoga', 'car', 'toothbrush', 'spoon', 'coffee_cup', 'sea_turtle', 'swan', 'key', 'eyeglasses', 'ceiling_fan', 'snail', 'clock', 'screwdriver', 'hockey_stick', 'lobster', 'school_bus', 'skyscraper', 'face', 'map', 'mailbox', 'radio', 'tiger', 'megaphone', 'lighter', 'book', 'cannon', 'cooler', 'sock', 'snorkel', 'bottlecap', 'sink', 'calculator', 'clarinet', 'zigzag', 'tree', 'mountain', 'birthday_cake', 'fan', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'banana', 'baseball', 'bed', 'table', 'scorpion', 'brain', 'train', 'toothpaste', 'bathtub', 'parachute', 'pond', 'onion', 'knee', 'mug', 'shovel', 'frog', 'flower', 'shoe', 'church', 'string_bean', 'shark', 'knife', 'pencil', 'cow', 'police_car', 'wine_bottle', 'goatee', 'bicycle', 'canoe', 'mushroom', 'cat', 'rain', 'suitcase', 'hot_dog', 'belt', 'hedgehog', 'cloud', 'camouflage', 'finger', 'hourglass', 'telephone', 'motorbike', 'tornado', 'dishwasher', 'microwave', 'skull', 'snowman', 'flying_saucer', 'marker', 'roller_coaster', 'lantern', 'mermaid', 'toaster', 'house_plant', 'octopus', 'snowflake', 'windmill', 'bird', 'horse']
        args.n_classes = 345
        args.n_domain = 6
    for domain in args.Domain_ID:
        if domain == args.source:
            # Single
            if SINGLE == True:
                args.target = args.Domain_ID.copy()
                args.target.remove(domain)
            else:
            # multiple 
                args.source = args.Domain_ID.copy()
                args.source.remove(domain)

            print("=" * 89)
            print("Training {} on source domains:")
            print(args.source)        
            print("Test on target domains:")
            print(args.target)

            main(args)
