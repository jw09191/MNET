import os
import yaml
import imageio
import argparse
import numpy as np

from tqdm import tqdm
from smplx import SMPL
from einops import rearrange, repeat
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

from src.utils.renderer import get_renderer
from src.models.components.models import DanceGenerator
from src.datamodules.components.dataset_utils import wav_processing, pkl_processing

import torch

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


def load_model(log_path, ckpt):
    path = os.path.join('./logs/experiments/runs/MNET', log_path)
    with open(os.path.join(path, '.hydra/config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ckpt = torch.load(os.path.join(path, 'checkpoints', ckpt), map_location='cpu')
    state_dict = {}
    for key, value in ckpt['state_dict'].items():
        key = key.split('.')
        if key[0] == 'gen':
            state_dict.update({'.'.join(key[1:]): value})

    model = DanceGenerator(**config['model']['gen_params'])
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_data(pkl_data, second, seed_m_length):
    pkl_data_path = os.path.join('./data/AIST++/motions', pkl_data)

    pose, trans = pkl_processing(pkl_data_path)
    motion = torch.cat([pose, trans], dim=1)

    audio_name = pkl_data.split('.')[0].split('_')[4]
    audio_path = os.path.join('./data/AIST++/wav', audio_name + '.wav')
    audio = wav_processing(audio_path, audio_name)

    genre_label = pkl_data.split('.')[0].split('_')[0]
    genre = torch.tensor(Genres[genre_label])

    audio = audio[:second * 60 + seed_m_length]
    seed_motion = motion[:seed_m_length]

    return audio, seed_motion, genre, audio_path


def render_video(motion, smpl, save_path, audio_path):
    width = 1024
    height = 1024

    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height)

    for idx, motion_ in enumerate(motion):
        save_name = os.path.join(save_path, f'z{idx}.mp4')
        writer = imageio.get_writer(save_name, fps=60)

        pose, trans = motion_[:, :-3].view(-1, 24, 3), motion_[:, -3:]
        meshes = smpl.forward(
            global_orient=pose[:, 0:1],
            body_pose=pose[:, 1:],
            transl=trans
        ).vertices.cpu().numpy()
        faces = smpl.faces

        meshes = meshes - meshes[0].mean(axis=0)
        cam = (0.55, 0.55, 0, 0.10)
        color = (0.2, 0.6, 1.0)

        imgs = []
        for ii, mesh in enumerate(tqdm(meshes, desc=f"Visualize dance - z{idx}")):
            img = renderer.render(background, mesh, faces, cam, color=color)
            imgs.append(img)

        imgs = np.array(imgs)
        for cimg in imgs:
            writer.append_data(cimg)
        writer.close()

        video_with_music(save_name, audio_path)


def video_with_music(save_video, audio_path):
    videoclip = VideoFileClip(save_video)
    audioclip = AudioFileClip(audio_path)

    if os.path.isfile(save_video):
        os.remove(save_video)

    new_audioclip = CompositeAudioClip([audioclip])
    new_audioclip = new_audioclip.cutout(videoclip.duration, audioclip.duration)

    videoclip.audio = new_audioclip
    videoclip.write_videofile(save_video, logger=None)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.log_path, args.ckpt)
    audio, seed_motion, genre, audio_path = load_data(args.pkl_data, args.second, model.seed_m_length)

    smpl = SMPL(model_path='./data/SMPL_DIR', gender='MALE', batch_size=1).eval()

    save_path = os.path.join('./logs/experiments/demos', args.log_path, args.type)
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    smpl = smpl.to(device)

    if args.type == 'diversity':
        num_sample = 5
        noise = torch.randn(num_sample, 256).to(device)
        genre = repeat(genre[None], '() -> b', b=num_sample).to(device)

    else:
        num_sample = 9
        noise = torch.randn(1, 256).to(device)
        noise = repeat(noise, '() d -> b d', b=num_sample).to(device)
        genre = [idx for idx in range(10) if idx != genre]
        genre = torch.tensor(genre).long().to(device)

    audio = repeat(audio[None], '() n d -> b n d', b=num_sample).to(device)
    seed_motion = repeat(seed_motion[None], '() n d -> b n d', b=num_sample).to(device)

    with torch.no_grad():
        output_motion = model.inference(audio, seed_motion, noise, genre)
        render_video(output_motion, smpl, save_path, audio_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="A Brand New Dance Partner")
    args.add_argument('-l', '--log_path', type=str, required=True)
    args.add_argument('-p', '--pkl_data', type=str, required=True)

    args.add_argument('-c', '--ckpt', type=str, default='last.ckpt')
    args.add_argument('-t', '--type', type=str, default='diversity')
    args.add_argument('-d', '--device', type=str, default='cuda:2')
    args.add_argument('-s', '--second', type=int, default=10)
    args = args.parse_args()

    main(args)
