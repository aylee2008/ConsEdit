import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pytorch_fid.fid_score import calculate_fid_given_paths
import lpips
from utils.io import load
import torch
import os
import torchvision.transforms as T
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from shutil import copyfile

from meta.coco import coco_classes, coco_dicts_rev
from meta.prepare_dataset import TEST_CLASSES_PTH, TEST_IM_DIR, TEST_OBJ_DETECT_DIR, \
                                IP2P_TEST_OBJ_DETECT_DIR, IP2P_TEST_OBJ_DETECT_DIR_ORI, \
                                IP2P_OUTPUT_DIR, IP2P_OUTPUT_DIR_ORI, \
                                FILTERED_FOR_CSFID_LPIPS_IP2P, FILTERED_FOR_CSFID_LPIPS_OURS
import pickle

resize = lambda n:T.Compose([
    T.Resize(n),
    T.CenterCrop(n),
    T.ToTensor(),
])

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
    
    assert len(input_files) == len(output_files)
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
    
    assert len(input_files) == len(output_files)
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

def prepare_filtered_outputs(input_img_dir, output_img_dir, test_classes_path, \
             input_obj_det_dir, output_obj_det_dir, filtered_in_dir, filtered_out_dir):
    filtered_list = calc_soft_acc(input_img_dir, output_img_dir, test_classes_path, \
             input_obj_det_dir, output_obj_det_dir,  True)

    for filename in filtered_list:
        copyfile(input_img_dir + filename, filtered_in_dir + filename)
        copyfile(output_img_dir + filename, filtered_out_dir + filename)

def prepare_filtered_dataset(ip2p_scales, cons_scales):
    for ip2p_scale in ip2p_scales:
        scale_output_img_path = IP2P_OUTPUT_DIR_ORI + "scale_" + ip2p_scale + "/"
        yolov7_output_path = IP2P_OUTPUT_DIR_ORI + "yolov7_results_scale_" + ip2p_scale + "/labels/"

        os.makedirs(FILTERED_FOR_CSFID_LPIPS_IP2P + "/input/scale_" + ip2p_scale, exist_ok = True)
        os.makedirs(FILTERED_FOR_CSFID_LPIPS_IP2P + "/output/scale_" + ip2p_scale, exist_ok = True)

        prepare_filtered_outputs(TEST_IM_DIR, scale_output_img_path, TEST_CLASSES_PTH, \
                                TEST_OBJ_DETECT_DIR, yolov7_output_path, \
                                FILTERED_FOR_CSFID_LPIPS_IP2P + "/input/scale_" + ip2p_scale + "/", \
                                FILTERED_FOR_CSFID_LPIPS_IP2P + "/output/scale_" + ip2p_scale + "/")

    for cons_scale in cons_scales:
        scale_output_img_path = IP2P_OUTPUT_DIR + "scale_" + cons_scale + "/"
        yolov7_output_path = IP2P_OUTPUT_DIR + "yolov7_results_scale_" + cons_scale + "/labels/"

        os.makedirs(FILTERED_FOR_CSFID_LPIPS_OURS + "/input/scale_" + cons_scale, exist_ok = True)
        os.makedirs(FILTERED_FOR_CSFID_LPIPS_OURS + "/output/scale_" + cons_scale, exist_ok = True)

        prepare_filtered_outputs(TEST_IM_DIR, scale_output_img_path, TEST_CLASSES_PTH, \
                                TEST_OBJ_DETECT_DIR, yolov7_output_path, \
                                FILTERED_FOR_CSFID_LPIPS_OURS + "/input/scale_" + cons_scale + "/", \
                                FILTERED_FOR_CSFID_LPIPS_OURS + "/output/scale_" + cons_scale + "/")
            

def calc_metrics(
    ip2p_scales,
    cons_scales,
    scale_input_img_paths,
    scale_output_img_paths
    
):
    for ip2p_scale in ip2p_scales:
        # scale_input_img_path = FILTERED_FOR_CSFID_LPIPS_IP2P + "/input/scale_" + ori_scale + "/"
        # scale_output_img_path = FILTERED_FOR_CSFID_LPIPS_IP2P + "/output/scale_" + ori_scale + "/"

        scale_input_img_path = scale_input_img_paths["ip2p"][ip2p_scale]
        scale_output_img_path = scale_output_img_paths["ip2p"][ip2p_scale]

        mean_ori_lpips = calc_lpips(scale_input_img_path, scale_output_img_path, "cpu")
        print(f"mean_lpips: {mean_ori_lpips}")

        ori_csfid = calc_csfid(scale_input_img_path, scale_output_img_path, TEST_CLASSES_PTH, "cpu", "_per_class_"+ip2p_scale+"/")
        print(f"csfid: {ori_csfid}")

        tough_acc = calc_tough_acc(scale_input_img_path, scale_output_img_path, \
                       TEST_CLASSES_PTH, TEST_OBJ_DETECT_DIR, \
                       IP2P_TEST_OBJ_DETECT_DIR_ORI[:-1] + "_scale_" + ip2p_scale + "/labels/")
        print(f"tough_acc: {tough_acc}")
        
        soft_acc = calc_soft_acc(scale_input_img_path, scale_output_img_path, \
                       TEST_CLASSES_PTH, TEST_OBJ_DETECT_DIR, \
                       IP2P_TEST_OBJ_DETECT_DIR_ORI[:-1] + "_scale_" + ip2p_scale + "/labels/")
        print(f"soft_acc: {soft_acc}")

        scale_res_dict = {
            "lpips": mean_ori_lpips,
            "csfid": ori_csfid,
            "tough_acc": tough_acc,
            "soft_acc": soft_acc
        }

        with open(scale_output_img_path[:-1] + ".pickle", "wb") as f:
            pickle.dump(scale_res_dict, f)

    for cons_scale in cons_scales:
        # scale_input_img_path = FILTERED_FOR_CSFID_LPIPS_OURS + "/input/scale_" + cons_scale + "/"
        # scale_output_img_path = FILTERED_FOR_CSFID_LPIPS_OURS + "/output/scale_" + cons_scale + "/"

        scale_input_img_path = scale_input_img_paths["cons"][cons_scale]
        scale_output_img_path = scale_output_img_paths["cons"][cons_scale]

        mean_our_lpips = calc_lpips(scale_input_img_path, scale_output_img_path, "cpu")
        print(f"mean_lpips: {mean_our_lpips}")

        our_csfid = calc_csfid(scale_input_img_path, scale_output_img_path, TEST_CLASSES_PTH, "cpu", "_per_class_"+cons_scale+"/")
        print(f"csfid: {our_csfid}")

        tough_acc = calc_tough_acc(TEST_IM_DIR, scale_output_img_path, \
                       TEST_CLASSES_PTH, TEST_OBJ_DETECT_DIR, \
                       IP2P_TEST_OBJ_DETECT_DIR[:-1] + "_scale_" + cons_scale + "/labels/")
        print(f"tough_acc: {tough_acc}")
        
        soft_acc = calc_soft_acc(TEST_IM_DIR, scale_output_img_path, \
                       TEST_CLASSES_PTH, TEST_OBJ_DETECT_DIR, \
                       IP2P_TEST_OBJ_DETECT_DIR[:-1] + "_scale_" + cons_scale + "/labels/")
        print(f"soft_acc: {soft_acc}")

        scale_res_dict = {
            "lpips": mean_our_lpips,
            "csfid": our_csfid,
            "tough_acc": tough_acc,
            "soft_acc": soft_acc
        }

        with open(scale_output_img_path[:-1] + ".pickle", "wb") as f:
            pickle.dump(scale_res_dict, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4)
    opt = parser.parse_args()

    ip2p_scales = ["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2"]
    cons_scales = ["0.2", "0.6", "1.0", "1.4", "1.8", "2.2", "2.6", "3.0", "3.4", "3.8"]
    tot_scale_input_img_paths = {
        False: {
            "ip2p": {
                ip2p_scale: TEST_IM_DIR for ip2p_scale in ip2p_scales
            },
            "cons": {
                cons_scale: TEST_IM_DIR for cons_scale in cons_scales
            }
        },
        True: {
            "ip2p": {
                ip2p_scale: FILTERED_FOR_CSFID_LPIPS_IP2P + "/input/scale_" + \
                ip2p_scale + "/" for ip2p_scale in ip2p_scales
            },
            "cons": {
                cons_scale: FILTERED_FOR_CSFID_LPIPS_OURS + "/input/scale_" + \
                cons_scale + "/" for cons_scale in cons_scales
            }
        }
    }
    tot_scale_output_img_paths = {
        False: {
            "ip2p": {
                ip2p_scale: IP2P_OUTPUT_DIR_ORI + "scale_" + ip2p_scale + "/" \
                for ip2p_scale in ip2p_scales
            },
            "cons": {
                cons_scale: IP2P_OUTPUT_DIR + "scale_" + cons_scale + "/" \
                for cons_scale in cons_scales
            }
        },
        True: {
            "ip2p": {
                ip2p_scale: FILTERED_FOR_CSFID_LPIPS_IP2P + "/output/scale_" + \
                ip2p_scale + "/" for ip2p_scale in ip2p_scales
            },
            "cons": {
                cons_scale: FILTERED_FOR_CSFID_LPIPS_OURS + "/output/scale_" + \
                cons_scale + "/" for cons_scale in cons_scales
            }
        }
    }

    # 0. prepare dataset (can skip this when only calculate non-filtered metric)
    prepare_filtered_dataset(ip2p_scales, cons_scales)

    # 1. calculate metrics
    for filtered in [False, True]:
        calc_metrics(ip2p_scales, cons_scales, tot_scale_input_img_paths[filtered], \
                     tot_scale_output_img_paths[filtered])