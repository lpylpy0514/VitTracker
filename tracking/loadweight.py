import torch
from lib.models.ostrack import build_ostrack
import argparse
import importlib
import pandas
import numpy as np
import cv2 as cv
import os
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
from segment_anything import sam_model_registry, SamPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default='ostrack', help='Name of the train script.')
    parser.add_argument('--config', type=str, default='experiments/ostrack/vitb_td_osckpt1.yaml', help="Name of the config file.")
    args = parser.parse_args()

    config_module = importlib.import_module("lib.config.%s.config" % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(args.config)
    model = build_ostrack(cfg)
    checkpoint = torch.load("output/checkpoints/train/ostrack/vitb_td_osckpt1/OSTrack_ep0300.pth.tar", map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    # sam_checkpoint = "sam_ckpt/sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # device = "cuda"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # predictor = SamPredictor(sam)

    for i in range(1, 180):
        data_dir = f"/home/ymz/newdisk1/GOT10k/train/GOT-10k_Train_{i:06}"
        gt_dir = data_dir + "/groundtruth.txt"
        gt = pandas.read_csv(gt_dir, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values

        imgs = []
        for file in os.listdir(data_dir):
            if file.endswith('jpg'):
                imgs.append(file)
        img_num = len(imgs)
        n = 1
        img = cv.imread(data_dir + f'/{n:08}.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        x1 = gt[0][0]
        y1 = gt[0][1]
        cx = int(gt[0][0] + 0.5 * gt[0][2])
        cy = int(gt[0][1] + 0.5 * gt[0][3])
        w = int(gt[0][2])
        h = int(gt[0][3])
        # x2, y2 = x1 + w, y1 + h
        # input_box = np.array([x1, y1, x2, y2])
        # predictor.set_image(img)
        #
        # masks, _, _ = predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_box[None, :],
        #     multimask_output=False,
        # )
        pre = Preprocessor()
        z_patch_arr, resize_factor, z_amask_arr = sample_target(img, [x1, y1, w, h], 2,
                                                                output_sz=128)  # (x1, y1, w, h)
        template = pre.process(z_patch_arr, z_amask_arr)
        model = model.cuda()
        color, transparency = model.template_preprocess.color.net(template.tensors.cuda())
        print(color)
        print(transparency)
        cv.imshow("template", z_patch_arr)
        cv.waitKey(0)
    cv.destroyAllWindows()