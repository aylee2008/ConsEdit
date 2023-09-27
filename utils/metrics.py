from pytorch_fid.fid_score import calculate_fid_given_paths
import lpips
from utils.io import load
import torch
import os
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
import pickle
from shutil import copyfile

resize = lambda n:T.Compose([
    T.Resize(n),
    T.CenterCrop(n),
    T.ToTensor(),
])

from meta.coco import coco_classes, coco_dicts_rev

def calc_csfid(input_img_dir, output_img_dir, test_classes_path, device, per_class_dir_postfix):
    # 0: classify image directories respect to image classes
    # inevitably get rid of "None" class (or erase instructions)
    input_img_files = sorted(os.listdir(input_img_dir))
    output_img_files = sorted(os.listdir(output_img_dir))
    with open(test_classes_path, "rb") as f:
        test_classes = pickle.load(f)

    input_per_class_files = dict()
    output_per_class_files = dict()

    #remove person class
    for curr_coco_class in coco_classes[1:]:
        input_per_class_files[curr_coco_class] = []
        output_per_class_files[curr_coco_class] = []

    for input_img_file, output_img_file in zip(input_img_files, output_img_files):
        assert input_img_file == output_img_file, "Two names should be equal!"
        if pd.isnull(test_classes[input_img_file][0]) or pd.isnull(test_classes[output_img_file][1]):
            continue

        input_per_class_files[test_classes[input_img_file][0]].append(input_img_file)
        output_per_class_files[test_classes[output_img_file][1]].append(output_img_file)

    for curr_coco_class in coco_classes[1:]:
        input_class_img_dir = input_img_dir[:-1] + "_per_class/" + f"{coco_dicts_rev[curr_coco_class]}/"
        output_class_img_dir = output_img_dir[:-1] + per_class_dir_postfix + f"{coco_dicts_rev[curr_coco_class]}/"
        os.makedirs(input_class_img_dir, exist_ok=True)
        os.makedirs(output_class_img_dir, exist_ok=True)

        # should only execute once!!
        for curr_inp_file in input_per_class_files[curr_coco_class]:
            copyfile(input_img_dir + curr_inp_file, input_class_img_dir + curr_inp_file)

        for curr_out_file in output_per_class_files[curr_coco_class]:
            copyfile(output_img_dir + curr_out_file, output_class_img_dir + curr_out_file)

    # 1: calculate csfid based on the classified directories
    # Current code only calculates fid
    # hardcoded values in https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    fid_val = []
    for curr_class_ind in tqdm(range(1, len(coco_classes))):
        input_class_img_dir = input_img_dir[:-1] + "_per_class/" + f"{curr_class_ind}/"
        output_class_img_dir = output_img_dir[:-1] + per_class_dir_postfix + f"{curr_class_ind}/"
        print(f"Length of input: {len(os.listdir(input_class_img_dir))}")
        print(f"Length of output: {len(os.listdir(output_class_img_dir))}")
        if len(os.listdir(input_class_img_dir)) == 0 or \
            len(os.listdir(output_class_img_dir)) == 0:
            print(f"Beware: passing {curr_class_ind}")
            continue
        
        try:
            fid_val.append(calculate_fid_given_paths([input_class_img_dir, \
                                             output_class_img_dir], batch_size=50,\
                                        device = device, dims=2048))
        except: #numerical instability
            pass
    
    print(f"Total length of fid_val is {len(fid_val)}")
    return sum(fid_val)/len(fid_val)

# referenced https://github.com/facebookresearch/SemanticImageTranslation/blob/main/eval.py
def calc_lpips(input_img_dir, output_img_dir, device, lpips_var=False, raw = False):

    input_files = sorted(os.listdir(input_img_dir))
    input_imgs = torch.stack([resize(256)(load(input_img_dir + x)) for x in input_files])
    output_files = sorted(os.listdir(output_img_dir))
    output_imgs = torch.stack([resize(256)(load(output_img_dir + x)) for x in output_files])

    lpnet = lpips.LPIPS(net='alex').to(device)
    lpips_dist = lambda x, y: lpnet.forward(x.to(device), 
                                            y.to(device), 
                                            normalize=True).detach().cpu()
    sim_scores = torch.cat([lpips_dist(output_img, input_img) for output_img, input_img
                            in zip(output_imgs.split(32), input_imgs.split(32))])
    mean_lpips = 100*sim_scores.mean().item()
    var_lpips = 100*sim_scores.std().item()

    if raw:
        return sim_scores

    if lpips_var:
        return (mean_lpips, var_lpips)
    else:
        return mean_lpips

def calc_tough_acc(input_img_dir, output_img_dir, test_classes_path, \
             input_obj_det_dir, output_obj_det_dir, get_filtered_list = False):
    input_files = sorted(os.listdir(input_img_dir))
    output_files = sorted(os.listdir(output_img_dir))
    
    if not len(input_files) == len(output_files):
        print("Warning: input files and output files are not equal. Returning zero instead. You can ignore this warning when processing filtering dataset")
        return 0
    
    tot_num = len(input_files)
    acc_num = 0

    create_num = 0
    erase_num = 0
    change_num = 0

    create_acc_num = 0
    erase_acc_num = 0
    change_acc_num = 0

    filtered_out_list = list()

    with open(test_classes_path, "rb") as f:
        test_classes = pickle.load(f)

    for input_file, output_file in zip(input_files, output_files):
        #print(f"Current input file: {input_file}")
        inp_ob_list = list()
        out_ob_list = list()
        assert input_file == output_file, "File name should be equal"

        input_class, output_class = test_classes[input_file]
        #print(f"input, output class: {input_class}, {output_class}")
        # print(f"input, output num: {coco_dicts_rev[input_class]}, {coco_dicts_rev[output_class]}")

        #print(input_obj_det_dir + input_file[:-4] + ".txt")
        try:
            with open(input_obj_det_dir + input_file[:-4] + ".txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    label = line.split(" ")[0]
                    inp_ob_list.append(label)
        except:
            pass

        try:
            with open(output_obj_det_dir + input_file[:-4] + ".txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    label = line.split(" ")[0]
                    out_ob_list.append(label)
        except:
            pass

        #print(f"inp list: {inp_ob_list}")
        #print(f"out list: {out_ob_list}")

        # create
        if pd.isnull(input_class) and not pd.isnull(output_class):
            inp_ob_list.append(coco_dicts_rev[output_class])
            create_num += 1
        # erase
        elif not pd.isnull(input_class) and pd.isnull(output_class):
            out_ob_list.append(coco_dicts_rev[input_class])
            erase_num += 1
        # change
        elif not pd.isnull(input_class) and not pd.isnull(output_class):
            inp_ob_list.append(coco_dicts_rev[output_class])
            out_ob_list.append(coco_dicts_rev[input_class])
            change_num += 1
        else:
            raise NotImplementedError
        
        inp_ob_list.sort()
        out_ob_list.sort()
        
        flag = 1 # 1 when accurate, 0 when inaccurate
        if len(inp_ob_list) == len(out_ob_list):
            for j in range(len(inp_ob_list)):
                if inp_ob_list[j] == out_ob_list[j]:
                    continue
                else:
                    flag = 0
        else:
            flag = 0

        if flag == 1:
            if pd.isnull(input_class) and not pd.isnull(output_class):
                create_acc_num += 1
            if not pd.isnull(input_class) and pd.isnull(output_class):
                erase_acc_num += 1
            if not pd.isnull(input_class) and not pd.isnull(output_class):
                change_acc_num += 1
            acc_num = acc_num + 1
            filtered_out_list.append(output_file)
        else:
            pass

    print(f"create acc: {create_acc_num}/{create_num}")
    print(f"erase acc: {erase_acc_num}/{erase_num}")
    print(f"change acc: {change_acc_num}/{change_num}")

    if get_filtered_list == True:
        return filtered_out_list
    else:
        return acc_num / tot_num

def calc_soft_acc(input_img_dir, output_img_dir, test_classes_path, \
             input_obj_det_dir, output_obj_det_dir, get_filtered_list = False):
    input_files = sorted(os.listdir(input_img_dir))
    output_files = sorted(os.listdir(output_img_dir))
    
    if not len(input_files) == len(output_files):
        print("Warning: input files and output files are not equal. Returning zero instead. You can ignore this warning when processing filtering dataset")
        return 0
    tot_num = len(input_files)
    acc_num = 0

    create_num = 0
    erase_num = 0
    change_num = 0

    create_acc_num = 0
    erase_acc_num = 0
    change_acc_num = 0

    filtered_out_list = list()

    with open(test_classes_path, "rb") as f:
        test_classes = pickle.load(f)

    for input_file, output_file in zip(input_files, output_files):
        #print(f"Current input file: {input_file}")
        inp_ob_list = list()
        out_ob_list = list()
        assert input_file == output_file, "File name should be equal"

        input_class, output_class = test_classes[input_file]
        #print(f"input, output class: {input_class}, {output_class}")
        # print(f"input, output num: {coco_dicts_rev[input_class]}, {coco_dicts_rev[output_class]}")

        #print(input_obj_det_dir + input_file[:-4] + ".txt")
        try:
            with open(input_obj_det_dir + input_file[:-4] + ".txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    label = line.split(" ")[0]
                    inp_ob_list.append(label)
        except:
            pass

        try:
            with open(output_obj_det_dir + input_file[:-4] + ".txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    label = line.split(" ")[0]
                    out_ob_list.append(label)
        except:
            pass

        #print(f"inp list: {inp_ob_list}")
        #print(f"out list: {out_ob_list}")

        flag = 1
        # create
        if pd.isnull(input_class) and not pd.isnull(output_class):
            if coco_dicts_rev[output_class] in out_ob_list:
                create_acc_num += 1
                pass
            else:
                flag = 0
            create_num += 1
        # erase
        elif not pd.isnull(input_class) and pd.isnull(output_class):
            if coco_dicts_rev[input_class] not in out_ob_list:
                erase_acc_num += 1
                pass
            else:
                flag = 0
            erase_num += 1
        # change
        elif not pd.isnull(input_class) and not pd.isnull(output_class):
            if coco_dicts_rev[input_class] not in out_ob_list and \
                coco_dicts_rev[output_class] in out_ob_list:
                change_acc_num += 1
                pass
            else:
                flag = 0
            change_num += 1
        else:
            raise NotImplementedError

        if flag == 1:
            acc_num = acc_num + 1
            filtered_out_list.append(output_file)

    print(f"create acc: {create_acc_num}/{create_num}")
    print(f"erase acc: {erase_acc_num}/{erase_num}")
    print(f"change acc: {change_acc_num}/{change_num}")

    if get_filtered_list == True:
        return filtered_out_list
    else:
        return acc_num / tot_num
