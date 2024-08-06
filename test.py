import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import util.misc as misc
from util.FSC147 import TTensor
from models_counting_network import CountingNetwork
import open_clip

import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.color import rgb2gray

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Testing Open-world Text-specified Object Counting Network"
    )

    parser.add_argument(
        "--data_split",
        default="val",
        help="data split of FSC-147 to test",
    )

    parser.add_argument(
        "--output_dir",
        default="./test",
        help="path where to save test log",
    )

    parser.add_argument("--device", default="cuda", help="device to use for testing")

    parser.add_argument(
        "--resume",
        default="./counting_network.pth",
        help="file name for model checkpoint to use for testing",
    )

    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_false",
        help="pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU",
    )

    parser.add_argument(
        "--img_dir",
        default="/scratch/local/hdd/nikian/images_384_VarV2",
        help="directory containing images from FSC-147",
    )

    parser.add_argument(
        "--FSC147_anno_file",
        default="/scratch/local/hdd/nikian/annotation_FSC147_384.json",
        help="name of file with FSC-147 annotations",
    )

    parser.add_argument(
        "--FSC147_D_anno_file",
        default="./FSC-147-D.json",
        help="name of file with FSC-147-D",
    )

    parser.add_argument(
        "--data_split_file",
        default="/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json",
        help="name of file with train, val, test splits of FSC-147",
    )

    return parser


open_clip_vit_b_16_preprocess = transforms.Compose(
    [
        transforms.Resize(
            size=224,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        ),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class TestData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split[args.data_split]

        with open(args.FSC147_anno_file) as f:
            fsc147_annotations = json.load(f)
        self.fsc147_annotations = fsc147_annotations

        with open(args.FSC147_D_anno_file) as f:
            fsc147_d_annotations = json.load(f)
        self.fsc147_d_annotations = fsc147_d_annotations

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        fsc147_anno = self.fsc147_annotations[im_id]
        fsc147_d_anno = self.fsc147_d_annotations[im_id]
        text = self.clip_tokenizer(fsc147_d_anno["text_description"]).squeeze(-2)

        dots = np.array(fsc147_anno["points"])

        image = Image.open("{}/{}".format(self.img_dir, im_id))
        image.load()
        W, H = image.size

        # This resizing step exists for consistency with CounTR's data resizing step.
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        image = transforms.Resize((new_H, new_W))(image)
        image = TTensor(image)

        return image, dots, text


def non_max_suppression(img, thresh):
    img_max_thresh = (img > (thresh * img.max()))
    return img_max_thresh * img


def main(args):

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Force PyTorch to be deterministic for reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html.
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    dataset_test = TestData(args)
    print(dataset_test)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Initialize the model.
    model = CountingNetwork()

    model.to(device)

    misc.load_model_FSC(args=args, model_without_ddp=model)

    print(f"Start testing.")
    start_time = time.time()

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Testing (" + args.data_split + ")"
    print_freq = 20

    test_mae = 0
    test_rmse = 0

    for data_iter_step, (samples, gt_dots, text_descriptions) in enumerate(
        metric_logger.log_every(data_loader_test, print_freq, header)
    ):

        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        text_descriptions = text_descriptions.to(device, non_blocking=True)

        _, _, h, w = samples.shape

        # Apply sliding window density map averaging technique used in CounTR.
        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < w:
                (output,) = model(
                    # open_clip_vit_b_16_preprocess(
                        samples[:, :, :, start : start + 384]
                    # )
                    ,
                    text_descriptions,
                )
                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0 : prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1 : 384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start : prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1 : w])

                density_map = (
                    density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                )

                prev = start + 383
                start = start + 128
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384

        # # Save the density map as an image
        # density_map_np = density_map.cpu().numpy()
        # density_map_np = (density_map_np / np.max(density_map_np) * 255).astype(np.uint8)

        # # You can specify a directory to save the density map images
        # output_dir = args.output_dir if args.output_dir else "density_maps"
        # Path(output_dir).mkdir(parents=True, exist_ok=True)

        # image_filename = os.path.join(output_dir, f'density_map_{data_iter_step}.png')
        # cv2.imwrite(image_filename, density_map_np)

        pred_cnt = torch.sum(density_map / 60).item()

        # compute and save images
        density_map = density_map.unsqueeze(0)

        # density_map = density_map / density_map.max()

        density_map_normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min())
        # # print(density_map_normalized)

        # # print(density_map_normalized.squeeze(0))
        # density_map_normalized = density_map
        # density_map_normalized = non_max_suppression(density_map_normalized.squeeze(0).cpu().numpy(), 0.1)
        # density_map_normalized = torch.tensor(density_map_normalized, device=device)
        # print(density_map_normalized)
        density_map_np = density_map_normalized.cpu().numpy()
        density_map_np = (density_map_np * 255).astype(np.uint8)
        density_map_np = np.squeeze(density_map_np)
        # density_map_colored = cv2.applyColorMap(density_map_np, cv2.COLORMAP_JET)
        density_map_colored = cv2.applyColorMap(density_map_np, cv2.COLORMAP_HOT)
        # density_map_colored = np.transpose(density_map_colored, (2, 0, 1))
        # density_map_colored = density_map_colored.astype(np.float32)
        # density_map_colored = density_map_colored / 255
        # density_map_colored = torch.tensor(density_map_colored, device=device)

        image_filename = os.path.join(args.output_dir, f'density_map_{data_iter_step}.png')
        cv2.imwrite(image_filename, density_map_colored)

        # density_map_colored = np.average(density_map_colored, axis=0)
        # density_map_colored = torch.tensor(density_map_colored, device=device)
        
        # Normalize and apply colormap to the density map
        # density_map = density_map / density_map.max()
        # density_map_colored = cm.jet(density_map.cpu().numpy()) # You can use other colormaps as well
        # density_map_grayscale = rgb2gray(density_map_colored[:,:,:,0:3])
        # density_map_grayscale = torch.tensor(density_map_grayscale, device=device)

        # Compute and save images
        # pred = density_map_colored
        # pred = density_map_normalized
        # pred = density_map

        # pred = torch.cat((pred, torch.zeros_like(pred), torch.zeros_like(pred)))
        # fig = samples[0] + pred / 2
        # fig = torch.clamp(fig, 0, 1)

        # pred_img = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
        # draw = ImageDraw.Draw(pred_img)
        # draw.text((w-50, h-50), str(round(pred_cnt)), (255, 255, 255))
        # pred_img = np.array(pred_img).transpose((2, 0, 1))
        # pred_img = torch.tensor(np.array(pred_img), device=device) + pred
        # full = torch.cat((samples[0], fig, pred_img), -1)

        # torchvision.utils.save_image(fig, (os.path.join(args.output_dir, f'vis_{data_iter_step}.png')))
        # torchvision.utils.save_image(pred_img, (os.path.join(args.output_dir, f'pred_{data_iter_step}.png')))
        # torchvision.utils.save_image(full, (os.path.join(args.output_dir, f'full_{data_iter_step}.png')))

        torchvision.utils.save_image(samples[0], (os.path.join(args.output_dir, f'full_{data_iter_step}.png')))

        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        test_mae += cnt_err
        test_rmse += cnt_err**2

        print(
            f"{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  AE: {cnt_err},  SE: {cnt_err ** 2} "
        )

    print("Averaged stats:", metric_logger)

    log_stats = {
        "MAE": test_mae / (len(data_loader_test)),
        "RMSE": (test_rmse / (len(data_loader_test))) ** 0.5,
    }

    print(
        "Test MAE: {:5.2f}, Test RMSE: {:5.2f} ".format(
            test_mae / (len(data_loader_test)),
            (test_rmse / (len(data_loader_test))) ** 0.5,
        )
    )

    with open(
        os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
