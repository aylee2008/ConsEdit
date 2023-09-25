import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import torch
import numpy as np
from einops import rearrange
import pickle
import argparse
from tqdm import tqdm

from clip_similarity import ClipSimilarity

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

np.random.seed(42)

from meta.prepare_dataset import IP2P_METRIC_DIR


def compute_metrics(inp_im_path,
                    gen_path,
                    prompts_path,
                    output_dir, 
                    curr_scale_img,
                    device):
    clip_similarity = ClipSimilarity().to(device)
    #clip_similarity = ClipSimilarity()

    sim_0_avg = 0
    sim_1_avg = 0
    sim_direction_avg = 0
    sim_image_avg = 0
    count = 0

    inp_im_files = sorted(os.listdir(inp_im_path))
    gen_imgs = sorted(os.listdir(gen_path))
    with open(prompts_path, "rb") as f:
        inp_out_prompts = pickle.load(f)

    # beware of '/'s.. assumed that curr_val_data_dir does not contain the following '/'.
    for curr_inp_img, gen_img in tqdm(zip(inp_im_files, gen_imgs)):
        # ex) 050000_20230234.jpg, 050000_20230234.jpg
        assert curr_inp_img == gen_img, "File name should be equal"

        input_prompt, output_prompt = inp_out_prompts[curr_inp_img]

        image_0 = Image.open(inp_im_path + curr_inp_img)
        reize_res = torch.randint(512, 513, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        gen = Image.open(gen_path + gen_img)
        reize_res = torch.randint(512, 513, ()).item()
        gen = gen.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        gen = rearrange(2 * torch.tensor(np.array(gen)).float() / 255 - 1, "h w c -> c h w")

        sim_0, sim_1, sim_direction, sim_image = clip_similarity(
            image_0[None].to(device), gen[None].to(device), [input_prompt], [output_prompt]
        )

        sim_0_avg += sim_0.item()
        sim_1_avg += sim_1.item()
        sim_direction_avg += sim_direction.item()
        sim_image_avg += sim_image.item()
        count += 1

    sim_0_avg /= count
    sim_1_avg /= count
    sim_direction_avg /= count
    sim_image_avg /= count

    result_dict = {
        "sim_0": sim_0_avg,
        "sim_1": sim_1_avg,
        "sim_direction": sim_direction_avg,
        "sim_image": sim_image_avg
    }

    # ex) output_dir/scale_1.0.pickle
    with open(output_dir + f"scale_{curr_scale_img}.pickle", "wb") as f:
        pickle.dump(result_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4)
    opt = parser.parse_args()

    gpu_to_scale = {
        0: "1.0",
        1: "1.2",
        2: "1.4",
        3: "1.6",
        4: "1.8",
        5: "2.0",
        6: "2.2"
    }

    compute_metrics(IP2P_METRIC_DIR + f"test_images/",
                    IP2P_METRIC_DIR + f"ip2p_outputs/scale_{gpu_to_scale[opt.gpu]}/",
                    IP2P_METRIC_DIR + "test_inp_out_prompts.pickle",
                    IP2P_METRIC_DIR + f"ip2p_outputs/",
                    gpu_to_scale[opt.gpu],
                    f"cuda:{opt.gpu}")
    
    compute_metrics(IP2P_METRIC_DIR + f"test_images/",
                    IP2P_METRIC_DIR + f"cons_outputs/scale_{gpu_to_scale[opt.gpu]}/",
                    IP2P_METRIC_DIR + "test_inp_out_prompts.pickle",
                    IP2P_METRIC_DIR + f"cons_outputs/",
                    gpu_to_scale[opt.gpu],
                    f"cuda:{opt.gpu}")


    