import argparse
import os
import os.path as osp
import sys

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt



def test(opt, seg, gmm, alias):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.to(device)

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)
            parse_raw = inputs['parse'].to(device)
            img_orig = inputs['img_orig'].to(device)
            hand_mask = inputs['hand_mask'].to(device) # [B, 1, H, W]
            parse_orig = inputs['parse_orig'].to(device) # [B, 20, H, W]

            # --- PART 1: Segmentation ---
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear', align_corners=False)
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear', align_corners=False)
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear', align_corners=False)
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear', align_corners=False)
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            parse_old.scatter_(1, parse_pred, 1.0)
            
            # Remapping standard labels
            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            # --- PART 2: Warping (GMM) ---
            # We use the prediction to guide the warp
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            
            # --- MULTISPECTRAL STRATEGY ---
            # Use "Blue" (Original Shirt) + "Brown" (Original Arms) as the Warp Target.
            # parse_orig is One-Hot [B, 20, H, W].
            # 5: Upper, 6: Dress, 7: Coat ("Blue")
            # 14: LArm, 15: RArm ("Brown")
            multispectral_target = (parse_orig[:, 5:6] + parse_orig[:, 6:7] + parse_orig[:, 7:8] + 
                                    parse_orig[:, 14:15] + parse_orig[:, 15:16]).clamp(0, 1)
                                    
            parse_cloth_gmm = F.interpolate(multispectral_target, size=(256, 192), mode='nearest')
            
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border', align_corners=False)
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border', align_corners=False)

            # --- PART 3: "MULTISPECTRAL" COMPOSITING ---
            
            # 1. Create the "Thermal" Hand Layers (High Priority)
            # FIX: Do NOT use parse labels 14/15 (Entire Arms) as forbidden zones!
            # Use the precise 'hand_mask' (Palm/Wrist only) that we passed from datasets.py.
            
            # "Thermal Map": This area is HOT. Cloth cannot exist here.
            # We interpolate hand_mask to the working resolution (256, 192) or just use it at full res later?
            # Here we are cutting warped_cm (256x192). So we need 256x192 mask.
            thermal_hand_map = F.interpolate(hand_mask, size=(256, 192), mode='bilinear', align_corners=False)
            
            # Note: We don't need 'hand_mask_agn' or 'hand_mask_raw' anymore.
            # just the Keypoint-based mask.
            
            # Upsample thermal map to full res for later? No, warped_cm is low res here?
            # Wait, line 122: warped_cm = F.grid_sample(cm, ...). cm is High Res?
            # cm was loaded in datasets.py as 'cm[key]' -> Resize(load_width). So it is 768x1024.
            # c is also 768x1024.
            # So warped_c is High Res.
            # So thermal_hand_map must be High Res (load_height, load_width).
            
            thermal_hand_map = F.interpolate(hand_mask, size=(opt.load_height, opt.load_width), mode='bilinear', align_corners=False)
            
            # 2. Hard-Cut the Warped Cloth using the Thermal Map
            warped_cm = warped_cm * (1.0 - thermal_hand_map)
            warped_c = warped_c * (1.0 - thermal_hand_map)

            # 3. Setup ALIAS Inputs
            # FIX: Use the PREDICTED segmentation (parse) which has the body inpainted, 
            # instead of parse_agnostic which has a hole (BG) at the torso.
            parse_agn_7ch = parse.clone()
            
            # Overwrite the 'Upper' channel (index 2) with the Warped Mask
            parse_agn_7ch[:, 2] = warped_cm[:, 0] 
            
            # Ensure BG channel (0) is removed where we placed the cloth
            parse_agn_7ch[:, 0] = parse_agn_7ch[:, 0] * (1 - warped_cm[:, 0])

            # Also fix upscale_mask to use PREDICTED labels (img preservation)
            # Index 1 is Paste/Face, Index 3 is Hair. (Based on test.py mapping lines 97-105)
            # Wait, line 99 says 1 is Paste (Face), line 101 says 3 is Hair.
            # So we preserve 1 and 3.
            # And Thermal Hand Map (Arms). 
            # Note: We do NOT preserve Index 2 (Upper) because checks revealed it's the target.


            # 4. Generate ALIAS Result
            misalign_mask = parse_agn_7ch[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse_agn_7ch, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse_agn_7ch, parse_div, misalign_mask)

            # --- PART 4: Final High-Res Composition ---
            output_hi = (output + 1) / 2.0
            
            # DEBUG: Save inputs to Verify SegGenerator failure
            if i == 0:
                 # Pose (Check if skeletal drawing is correct)
                 debug_pose = (pose + 1) / 2.0
                 torchvision.utils.save_image(debug_pose, os.path.join(opt.save_dir, opt.name, 'debug_pose.jpg'))
                 
                 # Cloth Reference (what we want to see)
                 torchvision.utils.save_image((c + 1)/2.0, os.path.join(opt.save_dir, opt.name, 'debug_ref_cloth.jpg'))

            # --- Robust Preservation Mask (Existing Logic) ---
            # FIX: parse_raw is ONE-HOT [B, 20, H, W].
            # FIX: Include 14, 15 (Arms) in "Old Cloth" to allow Long Sleeves to cover them.
            old_cloth_mask = (parse_raw[:, 5:6] + parse_raw[:, 6:7] + parse_raw[:, 7:8] + 
                              parse_raw[:, 14:15] + parse_raw[:, 15:16]).clamp(0, 1)
            
            # Preserve Everything ELSE + Explicitly Identify Hands
            preservation_mask = (1.0 - old_cloth_mask + hand_mask).clamp(0, 1)
            
            # Upscale mask logic...
            upscale_mask = preservation_mask * (1.0 - warped_cm[:, 0:1])
            
            # --- TEXTURE FUSION (Fix Color Quality) ---
            # Instead of relying solely on ALIAS (which might be blurry/desaturated), 
            # we composite the Raw Warped Cloth (warped_c) directly.
            # dense_warp_mask is where the warped cloth exists.
            dense_warp_mask = warped_cm[:, 0:1] # [B, 1, H, W]
            
            # Normalize warped_c to [0, 1]
            warped_c_norm = (warped_c + 1) / 2.0
            
            # Initial Composite: ALIAS Output + Original Preservation
            # FIX: img_orig is ALREADY [0, 1]
            img_orig_norm = img_orig 
            base_comp = output_hi * (1 - upscale_mask) + img_orig_norm * upscale_mask
            
            # Final Composite: Overlap Raw Warp on top (Texture Fusion)
            # We use dense_warp_mask to place the cloth.
            # Note: warped_c already has arm-holes cut (Step 2339).
            final_comp = base_comp * (1 - dense_warp_mask) + warped_c_norm * dense_warp_mask

            # Normalize for saving (Map [0, 1] back to [-1, 1])
            output_final = final_comp * 2.0 - 1.0

            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append('{}_{}'.format(img_name.split('_')[0], c_name))

            save_images(output_final, unpaired_names, os.path.join(opt.save_dir, opt.name))

            if (i + 1) % opt.display_freq == 0:
                print("step: {}".format(i + 1))


def main():
    opt = get_opt()
    print(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.to(device).eval()
    gmm.to(device).eval()
    alias.to(device).eval()
    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()