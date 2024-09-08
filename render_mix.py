import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image

def render_separate(source_path, source_name = 'emily', model_path = "output"):
    for component in ["spec", "diff", "all"]:

        path_s = os.path.join(source_path, source_name, component)
        path_m = os.path.join(model_path, source_name, component)

        cmd_train = 'python train.py -s ' + path_s + ' -m ' + path_m + ' --eval'
        os.system(cmd_train)
        for iter in [2000, 3000, 4000, 5000, 6000, 7000, 8000]:
            cmd_render = 'python render.py -m ' + path_m + " --iteration " + str(iter)
            os.system(cmd_render)


def mix_image(image_path, t, iter, file_name, output_path = "mixed", kd = 1.0, ks = 1.0):   
    
    spec_path = os.path.join(image_path, "spec", t, "ours_" + str(iter), "renders", file_name)
    diff_path = os.path.join(image_path, "diff", t, "ours_" + str(iter), "renders", file_name)

    # Open an image file
    image_spec = Image.open(spec_path)
    image_diff = Image.open(diff_path)

    if image_spec.size != image_diff.size:
        raise ValueError("The spec image and diff image must be of the same size")

    image_spec = image_spec.convert("RGB")
    image_diff = image_diff.convert("RGB")

    merged_image = Image.new('RGB', image_spec.size)

    # Load pixel values for two images, and load blank pixels for mix
    pixels_spec = image_spec.load()
    pixels_diff = image_diff.load()
    pixels_merged = merged_image.load()

    for y in range(image_spec.size[1]):
        for x in range(image_spec.size[0]):
            r1, g1, b1 = pixels_spec[x, y]
            r2, g2, b2 = pixels_diff[x, y]

            r = max(min(round(r1*ks + r2*kd), 255),0)
            g = max(min(round(g1*ks + g2*kd), 255),0)
            b = max(min(round(b1*ks + b2*kd), 255),0)

            # Set the pixel in the merged image
            pixels_merged[x, y] = (r, g, b)
    
    fdirectory = os.path.join(image_path, output_path, t, "ours_" + str(iter), "renders")
    if not os.path.exists(fdirectory):
        os.makedirs(fdirectory)
    fpath = os.path.join(fdirectory, file_name)
    merged_image.save(fpath)

    return merged_image

def mix_images(image_path, output_path = "mixed", kd = 1.0, ks = 1.0):
    for t in ["train","test"]:
        for iter in [2000, 3000, 4000, 5000, 6000, 7000, 8000]:
            spec_path = os.path.join(image_path, "spec", t, "ours_5000", "renders")
            for fname in os.listdir(spec_path):
                mix_image(image_path, t, iter, fname, output_path, kd, ks)
    return 


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Full rendering test parameters")
    parser.add_argument('--name',"-n", type=str, default="emily", help="The name of the dataset", required=True)
    parser.add_argument('--source-path',"-s", type=str, default="data", help="The name of the dataset")
    parser.add_argument('--model-path',"-m", type=str, default="output", help="The name of the dataset")
    parser.add_argument('--kd', type=float, default=1.0, required=False)
    parser.add_argument('--ks', type=float, default=1.0, required=False)
    args = parser.parse_args()

    print("Rendering dataset: " + args.name + "\n")
    render_separate(args.source_path, args.name, args.model_path)

    # Combine the shading components generated to form fully shaded output.
    mix_output_method  = "mixed"
    all_method = "all"
    gt_method = "gt"
    model_path = os.path.join("output", "emily-lower")
    print("Mixing diff and spec images")
    for ks in [1.0,0.8,0.7,0.6,0.5,0.4,0.3,0.2]:
        print("Mixing with ks ", ks)
        mix_images(model_path, mix_output_method + "-" + str(ks), 1.0, ks)


