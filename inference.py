import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import TOVO
import argparse

def enhance_single_image(model, image_path, output_path, iteration=None, device='cuda'):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor, iteration=iteration)

    save_image(output.squeeze(0), output_path)
    print(f"Saved enhanced image to {output_path}")

def enhance_folder(model, folder_path, output_folder, iteration=None, device='cuda'):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(folder_path, "*.*"))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        enhance_single_image(model, img_path, output_path, iteration=iteration, device=device)

def enhance_by_dataset(model, config, iteration=None, device='cuda'):
    data_path = config.dataset
    save_path = config.dataset
    if config.dataset == "LOL" or config.dataset == "LOLv2real" or config.dataset == "LOLv2synthetic":
        data_path = f"{data_path}/low"
        save_path = f"{save_path}/fake"
    image_path = f"{data_path}"  # Replace with actual image path
    save_dir = f"{config.model}/{save_path}"
    enhance_folder(model, image_path, save_dir, iteration=iteration, device=device)

def main():
    parser = argparse.ArgumentParser(description="Enhance low light images using TOVO model")
    parser.add_argument('--dataset', type=str, choices=['DICM', 'LIME', 'LOL', 'LOLv2synthetic', 'LOLv2real', 'MEF', 'NPE', 'VV', 'TRAIN', 'TEST', 'ExDark'], default='VV')
    parser.add_argument('--mode', type=str, choices=['single', 'folder', 'dataset'], default='single')
    parser.add_argument('--model', type=str, default="TOVO")
    parser.add_argument("--input", type=str, required=False, help="Path to image or folder")
    parser.add_argument("--output", type=str, required=False, help="Path to save enhanced image(s)")
    parser.add_argument("--iteration", type=int, default=5, help="Number of enhancement iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
    parser.add_argument("--color_space", type=str, default="hsv", help="Color space: hsv or rgb")
    args = parser.parse_args()

    model = TOVO(color_space=args.color_space, default_iteration=args.iteration)
    model.eval()
    model = model.to(args.device)

    if args.mode == 'folder':
        enhance_folder(model, args.input, args.output, iteration=args.iteration, device=args.device)
    elif args.mode == 'dataset':
        enhance_by_dataset(model, args, iteration=args.iteration, device=args.device)
    elif args.mode == 'single':
        enhance_single_image(model, args.input, args.output, iteration=args.iteration, device=args.device)
    else:
        print("Not supported mode!")
if __name__ == "__main__":
    main()
