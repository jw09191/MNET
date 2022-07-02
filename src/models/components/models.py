from src.models.components.layers import *
import src.utils.rotation_conversions as geometry


class DanceGenerator(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8, mlp_dim=2048,
                 music_length=240, seed_m_length=120, predict_length=20, rot_6d=True):
        super().__init__()
        self.music_length = music_length
        self.seed_m_length = seed_m_length
        self.predict_length = predict_length

        self.rot_6d = rot_6d

        self.mapping = MappingNet(256, dim)

        self.mlp_a = nn.Linear(438, dim)

        if rot_6d:
            self.mlp_m = nn.Linear(24 * 6 + 3, dim)
            self.mlp_l = nn.Linear(dim, 24 * 6 + 3)
        else:
            self.mlp_m = nn.Linear(24 * 3 + 3, dim)
            self.mlp_l = nn.Linear(dim, 24 * 3 + 3)

        self.tr_block = TransformerDecoder(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, audio, motion, noise, genre):
        if self.rot_6d:
            motion = geometry.matTOrot6d(motion)

        a = self.mlp_a(audio)
        m = self.mlp_m(motion)

        x = torch.cat([m, a], dim=1)
        s = self.mapping(noise, genre)[:, None]

        x = self.tr_block(x, s)
        x = self.mlp_l(x)[:, : self.predict_length]

        return x

    def inference(self, audio, motion, noise, genre):
        T = audio.shape[1]

        new_motion = motion
        for idx in range(0, T - self.music_length + 1, self.predict_length):
            audio_ = audio[:, idx:idx + self.music_length]

            motion_ = new_motion[:, -self.seed_m_length:]
            motion_ = self(audio_, motion_, noise, genre)

            if self.rot_6d:
                motion_ = geometry.rot6dTOmat(motion_)

            new_motion = torch.cat([new_motion, motion_], dim=1)
        return new_motion



class DanceDiscriminator(nn.Module):
    def __init__(self, dim=512, depth=4, heads=8, mlp_dim=2048, rot_6d=True):
        super().__init__()
        self.mlp_a = nn.Linear(438, dim)
        if rot_6d:
            self.mlp_m = nn.Linear(24 * 6 + 3, dim)
        else:
            self.mlp_m = nn.Linear(24 * 3 + 3, dim)

        self.mapping = MappingNet(dim, 1)
        self.tr_block = TransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, audio, motion, genre, genre_=None):
        a = self.mlp_a(audio)
        m = self.mlp_m(motion)

        x = torch.cat([m, a], dim=1)
        x = self.tr_block(x)

        x = x.mean(dim=1)
        if genre_ is None:
            return self.mapping(x, genre)
        else:
            return self.mapping(x, genre), self.mapping(x, genre_)


if __name__ == '__main__':
    ####################################################
    print("[*] Start DanceGeneratorAMNG_D")
    G = DanceGenerator(predict_length=30)

    audio = torch.randn(2, 240, 35)
    audioF = torch.randn(2, 60 * 11 + 90, 35)

    noise = torch.randn(2, 256)

    motion = torch.randn(2, 120, 24 * 3 + 3)
    genre = torch.tensor([5, 4])

    l_motion = G(audio, motion, noise, genre)

    motionD = G.diversity(audioF, motion, noise, genre)

    D = DanceDiscriminator()
    # l_motion = l_motion[:, :, :-3]
    logit = D(audio, l_motion, genre)


    print("[*] Finish DanceGeneratorAMNG_D")
    print("*******************************")

