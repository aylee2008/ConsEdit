import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import matplotlib.pyplot as plt

from meta.prepare_dataset import IP2P_METRIC_DIR, CONS_METRIC_DIR, \
                                IP2P_OUTPUT_DIR_ORI, IP2P_OUTPUT_DIR, \
                                FILTERED_FOR_CSFID_LPIPS_IP2P, FILTERED_FOR_CSFID_LPIPS_OURS

def plot_metric_ip2p(scale_list):

    plt.rc('font', size=15)        
    plt.rc('axes', labelsize=15)   
    plt.rc('xtick', labelsize=12)  
    plt.rc('ytick', labelsize=12)  
    plt.rc('legend', fontsize=15)  
    plt.rc('figure', titlesize=15)

    plt.figure(figsize=(6,5))

    results_dict_ori = dict()
    for scale in scale_list:
        with open(IP2P_METRIC_DIR + "ip2p_outputs/" + "scale_" + scale + ".pickle", "rb") as f:
            results_dict_ori[scale] = pickle.load(f)

    results_dict = dict()
    for scale in scale_list:
        with open(IP2P_METRIC_DIR + "cons_outputs/" + "scale_" + scale + ".pickle", "rb") as f:
            results_dict[scale] = pickle.load(f)

    res_dicts = [results_dict_ori, results_dict]
    res_labels = ["ip2p", "ours"]

    for i, res_dict in enumerate(res_dicts):
        sim_dir_res = list()
        sim_im_res = list()

        for key in res_dict.keys():
            sim_dir_res.append(res_dict[key]["sim_direction"])
            sim_im_res.append(res_dict[key]["sim_image"])
            
            # print(key)
            # print(res_dict[key]["sim_direction"])
            # print(res_dict[key]["sim_image"])

        plt.plot(sim_dir_res, sim_im_res, marker='o', label = res_labels[i])

    plt.legend()

    plt.xlabel("CLIP Direction Similarity")
    plt.ylabel("CLIP Image Similarity")

    plt.tight_layout()
    plt.savefig(IP2P_METRIC_DIR + "plot.png", dpi=400)
    
    print(f"plot finished!")

def plot_metric_cons(pickle_paths, ip2p_scales, cons_scales, filtered): #get_plots_and_results

    plt.rc('font', size=15)        
    plt.rc('axes', labelsize=15)   
    plt.rc('xtick', labelsize=15)  
    plt.rc('ytick', labelsize=15)  
    plt.rc('legend', fontsize=15)  
    plt.rc('figure', titlesize=15) 

    results_dict_ori = dict()
    for scale in ip2p_scales:
        with open(pickle_paths["ip2p"][scale], "rb") as f:
            results_dict_ori[scale] = pickle.load(f)

    results_dict = dict()
    for scale in cons_scales:
        with open(pickle_paths["cons"][scale], "rb") as f:
            results_dict[scale] = pickle.load(f)

    res_dicts = [results_dict_ori, results_dict]
    res_labels = ["ip2p", "ours"]

    # 0
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    for i, res_dict in enumerate(res_dicts):
        lpips_res = list()
        csfid_res = list()
        soft_acc_res = list()
        tough_acc_res = list()
        key_scales = list()
        for key in res_dict.keys():
            soft_acc = res_dict[key]["soft_acc"]
            tough_acc = res_dict[key]["tough_acc"]
            lpips_res.append(res_dict[key]["lpips"])
            csfid_res.append(res_dict[key]["csfid"])
            soft_acc_res.append(res_dict[key]["soft_acc"])
            tough_acc_res.append(res_dict[key]["tough_acc"])
            key_scales.append(key)

            print(f"Currently processing scale {key}")
            print(f"Soft accuracy: {soft_acc}")
            print(f"Tough accuracy: {tough_acc}")
            print(res_dict[key]["csfid"])
            print(res_dict[key]["lpips"])

        ax[0].plot(lpips_res, tough_acc_res, marker='o', label=res_labels[i])
        ax[1].plot(lpips_res, soft_acc_res, marker='o', label=res_labels[i])

    ax[0].set_title("acc-LPIPS")
    ax[0].set_ylabel("acc")
    ax[0].set_xlabel("LPIPS")
    ax[0].legend()

    ax[1].set_title("relaxed acc-LPIPS")
    ax[1].set_ylabel("relaxed acc")
    ax[1].set_xlabel("LPIPS")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(CONS_METRIC_DIR + f"plot_acc_{filtered}.png", dpi=400)

    plt.clf()

    # 1
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for i, res_dict in enumerate(res_dicts):
        lpips_res = list()
        csfid_res = list()
        key_scales = list()
        for key in res_dict.keys():
            lpips_res.append(res_dict[key]["lpips"])
            csfid_res.append(res_dict[key]["csfid"])
            key_scales.append(key)

            print(f"Currently processing scale {key}")
            print(res_dict[key]["csfid"])
            print(res_dict[key]["lpips"])

        ax.plot(lpips_res, csfid_res, marker='o', label=res_labels[i])

    ax.set_title("CSFID-LPIPS")
    ax.set_ylabel("CSFID")
    ax.set_xlabel("LPIPS")
    ax.legend()

    plt.tight_layout()
    plt.savefig(CONS_METRIC_DIR + f"plot_curve_{filtered}.png", dpi=400)
    
    print(f"plot finished!")

if __name__ == "__main__":
    plot_metric_ip2p(["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2"])

    ip2p_scales = ["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2"]
    cons_scales = ["0.2", "0.6", "1.0", "1.4", "1.8", "2.2", "2.6", "3.0", "3.4", "3.8"]

    tot_pickle_paths = {
        False: {
            "ip2p": {
                scale: IP2P_OUTPUT_DIR_ORI + "scale_" + scale + ".pickle" \
                for scale in ip2p_scales
            },
            "cons": {
                scale: IP2P_OUTPUT_DIR + "scale_" + scale + ".pickle" \
                for scale in cons_scales
            }
        },
        True: {
            "ip2p": {
                scale: FILTERED_FOR_CSFID_LPIPS_IP2P + "output/scale_" + scale + ".pickle" \
                for scale in ip2p_scales
            },
            "cons": {
                scale: FILTERED_FOR_CSFID_LPIPS_OURS + "output/scale_" + scale + ".pickle" \
                for scale in cons_scales
            }
        }
    }

    for filtered in [False, True]:
        plot_metric_cons(tot_pickle_paths[filtered], ip2p_scales, cons_scales, filtered)