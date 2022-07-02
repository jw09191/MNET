import os
import imageio
import numpy as np

from einops import rearrange, repeat

from tqdm import tqdm
from smplx import SMPL

import torch
from pytorch_lightning import Callback

import src.utils.rotation_conversions as geometry

from src.utils.renderer import get_renderer
from src.datamodules.components.dataset_utils import wav_processing, pkl_processing

Genres = {
    'gBR': 0,
    'gPO': 1,
    'gLO': 2,
    'gMH': 3,
    'gLH': 4,
    'gHO': 5,
    'gWA': 6,
    'gKR': 7,
    'gJS': 8,
    'gJB': 9,
}


class DanceGeneration(Callback):
    def __init__(self, data_path, data_name, seed_m_length, play_time=20, num_sample=10, smpl_path="../../../SMPL_DIR"):
        self.seed_m_length = seed_m_length

        self.play_time = play_time
        self.num_sample = num_sample

        self.smpl = SMPL(model_path=smpl_path, gender='MALE', batch_size=1).eval()

        self.base_genre_idx = Genres[data_name.split('.')[0].split('_')[0]]
        self.synt_genre_idx = np.setdiff1d(np.array(range(len(Genres))), self.base_genre_idx)

        motion_path = os.path.join(data_path, 'motions', data_name)
        pose, trans = pkl_processing(motion_path)
        self.motion = torch.cat([pose, trans], dim=1)[:self.seed_m_length]

        audio_name = data_name.split('.')[0].split('_')[4]
        audio_path = os.path.join(data_path, 'wav', audio_name + '.wav')
        self.audio = wav_processing(audio_path, audio_name)

        self.save_path = './callback_render_video'
        os.makedirs(self.save_path, exist_ok=True)

        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            self.diversity(pl_module)

    def diversity(self, pl_module):
        T = self.play_time * 60 + self.seed_m_length

        audio = repeat(self.audio[None, :T], '() n d -> b n d', b=self.num_sample).to(pl_module.device)
        seed_motion = repeat(self.motion[None], '() n d -> b n d', b=self.num_sample).to(pl_module.device)

        seed_genre = torch.tensor([self.base_genre_idx])
        seed_genre = repeat(seed_genre, '() -> b', b=self.num_sample).to(pl_module.device)

        noise = torch.randn(self.num_sample, 256).to(pl_module.device)

        with torch.no_grad():
            output_motion = pl_module.gen.diversity(audio, seed_motion, noise, seed_genre)
            self.render_video(output_motion, pl_module, "diversity")

    def render_video(self, motion, pl_module, state):
        smpl = self.smpl.to(device=pl_module.device)

        width = 1024
        height = 1024

        background = np.zeros((height, width, 3))
        renderer = get_renderer(width, height)

        for ii, motion_ in enumerate(motion):
            save_path = os.path.join(self.save_path, f'epoch{pl_module.current_epoch}_{state}_z{ii}.mp4')
            writer = imageio.get_writer(save_path, fps=60)

            pose, trans = motion_[:, :-3].view(-1, 24, 3), motion_[:, -3:]
            # smpl_poses = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))

            smpl_result = smpl.forward(
                global_orient=pose[:, 0:1],
                body_pose=pose[:, 1:],
                transl=trans)

            meshes = smpl_result.vertices.cpu().numpy()
            # joints = smpl_result.joints.cpu().numpy()
            faces = smpl.faces

            # center the first frame
            meshes = meshes - meshes[0].mean(axis=0)
            cam = (0.55, 0.55, 0, 0.10)
            color = (0.2, 0.6, 1.0)

            imgs = []
            for jj, mesh in enumerate(tqdm(meshes, desc=f"Visualize dance - {state}_z{ii}")):
                img = renderer.render(background, mesh, faces, cam, color=color)
                imgs.append(img)

            imgs = np.array(imgs)
            for cimg in imgs:
                writer.append_data(cimg)
            writer.close()



