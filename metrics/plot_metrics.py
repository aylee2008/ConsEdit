import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import matplotlib.pyplot as plt

from meta.prepare_dataset import IP2P_METRIC_DIR

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

if __name__ == "__main__":
    plot_metric_ip2p(["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2"])