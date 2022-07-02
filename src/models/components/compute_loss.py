import torch
import torch.nn as nn

from smplx import SMPL
from einops import rearrange
import src.utils.rotation_conversions as geometry


class ComputeLoss(nn.Module):
    def __init__(self, loss_dict, rot_6d, smpl_path):
        super(ComputeLoss, self).__init__()
        self.rot_6d = rot_6d

        self.gen_loss_dict = loss_dict
        self.dis_loss_dict = {}

        for loss_name in loss_dict.keys():
            if loss_name == 'l_rec_loss' or 'l_smt_loss':
                self.l2_loss = nn.MSELoss()
            if loss_name == 'l_vet_loss':
                self.vet_loss = VerticesLoss(smpl_path, rot_6d)
            if loss_name == 'l_div_loss':
                self.l1_loss = nn.L1Loss()
            if loss_name == 'l_adv_loss':
                self.adv_loss = AdversarialLoss('hinge')
                self.dis_loss_dict.update({loss_name: loss_dict[loss_name]})

    def __call__(self, **kwargs):
        state = kwargs['state']
        if state in ["train_gen", "valid"]:
            return self.gen_loss(**kwargs)
        elif state in ["train_dis"]:
            return self.dis_loss(**kwargs)

    def gen_loss(self, **kwargs):
        state = kwargs['state']
        music, dance_s, dance_f, dance_id = kwargs['music'], kwargs['dance_s'], kwargs['dance_f'], kwargs['dance_id']

        b = music.shape[0]
        device = music.device

        noise = torch.randn(b, 256).to(device)
        dance_g = kwargs['gen'](music, dance_s, noise, dance_id)

        if self.rot_6d:
            dance_f = geometry.matTOrot6d(dance_f)

        loss, log_dict = 0.0, {}
        for key, value in self.gen_loss_dict.items():
            if key == 'l_adv_loss':
                f_logit = kwargs['dis'](music, dance_g, dance_id)
                l_adv_loss = self.adv_loss(f_logit, True, False)
                loss += (l_adv_loss * value)

                log_dict.update({f'{state}/l_adv_loss': l_adv_loss})

            elif key == 'l_rec_loss':
                l_rec_loss = self.l2_loss(dance_f, dance_g)
                loss += (l_rec_loss * value)

                log_dict.update({f'{state}/l_rec_loss': l_rec_loss})

            elif key == 'l_vet_loss':
                l_vet_loss = self.vet_loss(dance_f, dance_g)
                loss += (l_vet_loss * value)

                log_dict.update({f'{state}/l_vet_loss': l_vet_loss})

            elif key == 'l_div_loss':
                noise1 = torch.randn(b, 256).to(device)
                noise2 = torch.randn(b, 256).to(device)

                style1 = kwargs['gen'].mapping(noise1, dance_id)
                style2 = kwargs['gen'].mapping(noise2, dance_id)

                l_div_loss = self.l1_loss(noise1, noise2) / self.l1_loss(style1, style2)
                loss += (l_div_loss * value)

                log_dict.update({f'{state}/l_div_loss': l_div_loss})

            elif key == 'l_sfc_loss':
                g_tensor = torch.arange(10).long().to(device)
                dance_id_ = [g_tensor[g_tensor != id][torch.randperm(9)[0]] for id in dance_id]
                dance_id_ = torch.stack(dance_id_, dim=0)

                dance_g_sfc = kwargs['gen'](music, dance_s, noise, dance_id_)
                r_logit, f_logit = kwargs['dis'](music, dance_g_sfc, dance_id_, dance_id)

                l_sfc_real = self.adv_loss(r_logit, True, False)
                l_sfc_fake = self.adv_loss(f_logit, False, False)

                l_sfc_loss = l_sfc_real + l_sfc_fake
                loss += (l_sfc_loss * value)

                log_dict.update({f'{state}/l_sfc_loss': l_sfc_loss})

        log_dict.update({f'{state}/loss': loss})
        return loss, log_dict

    def dis_loss(self, **kwargs):
        state = kwargs['state']
        music, dance_s, dance_f, dance_id = kwargs['music'], kwargs['dance_s'], kwargs['dance_f'], kwargs['dance_id']

        b = music.shape[0]
        device = music.device

        noise = torch.randn(b, 256).to(device)
        dance_g = kwargs['gen'](music, dance_s, noise, dance_id)

        if kwargs['gen'].rot_6d:
            pose, trans = dance_f[:, :, :-3], dance_f[:, :, -3:]
            pose = rearrange(pose, 'b t (d c) -> (b t) d c', d=24, c=3)
            pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
            pose = rearrange(pose, '(b t) d c -> b t (d c)', b=trans.shape[0], t=trans.shape[1])
            dance_f = torch.cat([pose, trans], dim=2)

        loss, log_dict = 0.0, {}
        for key, value in self.dis_loss_dict.items():
            if key == 'l_adv_loss':
                r_logit = kwargs['dis'](music, dance_f, dance_id)
                f_logit = kwargs['dis'](music, dance_g.detach(), dance_id)

                l_dis_real = self.adv_loss(r_logit, True, True)
                l_dis_fake = self.adv_loss(f_logit, False, True)

                l_adv_loss = ((l_dis_real + l_dis_fake) / 2)
                loss += (l_adv_loss * value)

                log_dict.update({f'{state}/l_real_loss': l_dis_real})
                log_dict.update({f'{state}/l_fake_loss': l_dis_fake})

        log_dict.update({f'{state}/loss': loss})
        return loss, log_dict


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type="nsgan"):
        """ type = nsgan | lsgan | hinge """
        super(AdversarialLoss, self).__init__()
        self.type = type
        if type == "nsgan":
            self.criterion = nn.BCELoss()
        elif type == "lsgan":
            self.criterion = nn.MSELoss()
        elif type == "hinge":
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == "hinge":
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = torch.ones_like(outputs) if is_real else torch.zeros_like(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class VerticesLoss(nn.Module):
    def __init__(self, model_path, rot_6d):
        super(VerticesLoss, self).__init__()
        self.smpl = SMPL(model_path=model_path, gender='MALE', batch_size=1).eval()
        self.rec_loss = nn.MSELoss()
        self.rot_6d = rot_6d

    def __call__(self, target, output):
        l_vet_loss = []

        if self.rot_6d:
            target = geometry.rot6dTOmat(target)
            output = geometry.rot6dTOmat(output)

        for tar, out in zip(target, output):
            pose, trans = tar[:, :-3].view(-1, 24, 3), tar[:, -3:]
            pose_, trans_ = out[:, :-3].view(-1, 24, 3), out[:, -3:]

            # smpl_poses = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))
            # smpl_poses_ = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose_))

            with torch.no_grad():
                vertices = self.smpl.forward(
                    global_orient=pose[:, 0:1],
                    body_pose=pose[:, 1:],
                    transl=trans
                ).vertices

                vertices_ = self.smpl.forward(
                    global_orient=pose_[:, 0:1],
                    body_pose=pose_[:, 1:],
                    transl=trans_
                ).vertices

                l_vet_loss.append(self.rec_loss(vertices, vertices_))

        l_vet_loss = torch.mean(torch.stack(l_vet_loss))
        return l_vet_loss
