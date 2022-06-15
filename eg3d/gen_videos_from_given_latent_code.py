# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator

# ----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


# ----------------------------------------------------------------------------

def gen_interp_video(G, latent, mp4: str,  w_frames=60 * 4, kind='cubic', grid_dims=(1, 1),
                     num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image',
                     gen_shapes=False, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    name = mp4[:-4]
    if num_keyframes is None:

        num_keyframes = 1 // (grid_w * grid_h)

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0],
                                                                                                      device=device)

    # zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(latent.shape[0], 1)
    ws = latent # 1, 14, 512

    if ws.shape[1] != G.backbone.mapping.num_ws:
        ws = ws.repeat([1,G.backbone.mapping.num_ws, 1])


    _ = G.synthesis(ws[:1], c[:1])  # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1]) # (5, 14, 512)
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)
    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = 'interpolation_{}/'.format(name)
        os.makedirs(outdir, exist_ok=True)
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    camera_lookat_point, radius=2.7, device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

                if gen_shapes:
                    # generate shapes
                    print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))

                    samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0],
                                                                       cube_length=G.rendering_kwargs['box_warp'])
                    samples = samples.to(device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm(total=samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                torch.manual_seed(0)
                                sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                                       transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                                       w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                sigmas[:, head:head + max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)

                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = 0
                    sigmas[-pad:] = 0
                    sigmas[:, :pad] = 0
                    sigmas[:, -pad_top:] = 0
                    sigmas[:, :, :pad] = 0
                    sigmas[:, :, -pad:] = 0

                    output_ply = True
                    if output_ply:
                        from shape_utils import convert_sdf_samples_to_ply
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                                   os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                    else:  # output mrc
                        with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape,
                                              mrc_mode=2) as mrc:
                            mrc.data[:] = sigmas

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)


# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--npy_path', 'npy_path', help='Network pickle filename', required=True)
@click.option('--num-keyframes', type=int,
              help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
              default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR',
              default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def generate_images(
        network_pkl: str,
        npy_path:str,
        truncation_psi: float,
        truncation_cutoff: int,
        num_keyframes: Optional[int],
        w_frames: int,
        outdir: str,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    if 'pkl' in network_pkl:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    else:
        init_args = ()
        init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
                       'channel_max': 512, 'fused_modconv_default': 'inference_only',
                       'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48,
                                            'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
                                            'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
                                            'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                                            'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                            'c_gen_conditioning_zero': False, 'c_scale': 1.0,
                                            'superresolution_noise_mode': 'none', 'density_reg': 0.25,
                                            'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                            'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
                       'sr_kwargs': {'channel_base': 32768, 'channel_max': 512,
                                     'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 25,
                       'img_resolution': 512, 'img_channels': 3}
        rendering_kwargs = {'depth_resolution': 96, 'depth_resolution_importance': 96, 'ray_start': 2.25,
                            'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
                            'image_resolution': 512, 'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                            'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                            'c_gen_conditioning_zero': False, 'c_scale': 1.0, 'superresolution_noise_mode': 'none',
                            'density_reg': 0.25, 'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                            'sr_antialias': True}

        # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
        print("Reloading Modules!")
        G = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(device)

        ckpt = torch.load(network_pkl)
        G.load_state_dict(ckpt['G_ema'], strict=False)
        G.neural_rendering_resolution = 128

        G.rendering_kwargs = rendering_kwargs

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    grid = (1,1)
    latent  = np.load(npy_path)
    latent = torch.FloatTensor(latent).cuda()
    name = os.path.basename(npy_path)[:-4]
    output = os.path.join(outdir, f'{name}.mp4')
    print('generate ',output)
    gen_interp_video(G=G, latent = latent,mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes,
                     w_frames=w_frames,  psi=truncation_psi,
                     truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode)



# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------