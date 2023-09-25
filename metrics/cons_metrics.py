import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import argparse
import pickle
from shutil import copyfile
from meta.prepare_dataset import TEST_CLASSES_PTH, TEST_IM_DIR, TEST_OBJ_DETECT_DIR, \
                                IP2P_TEST_OBJ_DETECT_DIR, IP2P_TEST_OBJ_DETECT_DIR_ORI, \
                                IP2P_OUTPUT_DIR, IP2P_OUTPUT_DIR_ORI, \
                                FILTERED_FOR_CSFID_LPIPS_IP2P, FILTERED_FOR_CSFID_LPIPS_OURS
from utils.metrics import calc_csfid, calc_lpips, calc_soft_acc, calc_tough_acc


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