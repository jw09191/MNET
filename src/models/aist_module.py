from src.models.components.compute_loss import ComputeLoss
from src.models.components.models import DanceGenerator, DanceDiscriminator

import torch
from pytorch_lightning import LightningModule


class AISTLitModule(LightningModule):
    def __init__(self, gen_params, dis_params, loss_params, optimizer, scheduler):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        assert gen_params, 'No generator parameters'
        self.gen = DanceGenerator(**gen_params)
        self.dis = DanceDiscriminator(**dis_params) if dis_params else None

        if self.gen and self.dis:
            if "l_adv_loss" not in loss_params.loss_dict.keys():
                loss_params.loss_dict.update({"l_adv_loss": 1})
        else:
            if "l_adv_loss" in loss_params.loss_dict.keys():
                loss_params.loss_dict.pop("l_adv_loss")

        # loss function
        self.compute_loss = ComputeLoss(**loss_params)

    def training_step(self, batch, batch_idx):

        if self.dis:
            gen_optim, dis_optim = self.optimizers()

            gen_optim.zero_grad()
            gen_loss, gen_log_dict = self.gen_step(batch, "train_gen")

            self.log_dict(gen_log_dict, on_step=True, on_epoch=False, prog_bar=False)
            self.manual_backward(gen_loss)
            gen_optim.step()

            dis_optim.zero_grad()
            dis_loss, dis_log_dict = self.dis_step(batch, "train_dis")

            self.log_dict(dis_log_dict, on_step=True, on_epoch=False, prog_bar=False)
            self.manual_backward(dis_loss)
            dis_optim.step()

            if self.hparams.scheduler:
                gen_scheduler, dis_scheduler = self.lr_schedulers()
                gen_scheduler.step()
                dis_scheduler.step()

        else:
            gen_optim = self.optimizers()

            gen_optim.zero_grad()
            loss, log_dict = self.gen_step(batch, "train_gen")

            self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)
            self.manual_backward(loss)
            gen_optim.step()

            if self.hparams.scheduler:
                gen_scheduler = self.lr_schedulers()
                gen_scheduler.step()

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.gen_step(batch, "valid")
        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def gen_step(self, batch, state):
        dance_s, dance_f, music, dance_id = batch
        loss_op_dict = {'state': state,
                        'music': music,
                        'dance_s': dance_s,
                        'dance_f': dance_f,
                        'dance_id': dance_id,
                        'gen': self.gen}
        if self.dis:
            loss_op_dict.update({'dis': self.dis})
        return self.compute_loss(**loss_op_dict)

    def dis_step(self, batch, state):
        dance_s, dance_f, music, dance_id = batch
        loss_op_dict = {'state': state,
                        'music': music,
                        'dance_s': dance_s,
                        'dance_f': dance_f,
                        'dance_id': dance_id,
                        'gen': self.gen,
                        'dis': self.dis}
        return self.compute_loss(**loss_op_dict)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer
        scheduler = self.hparams.scheduler

        if self.dis:
            gen_optim = getattr(torch.optim, optimizer['type'])(self.gen.parameters(), **optimizer['kwargs'])
            dis_optim = getattr(torch.optim, optimizer['type'])(self.dis.parameters(), **optimizer['kwargs'])
            if scheduler:
                gen_schedular = getattr(torch.optim.lr_scheduler, scheduler['type'])(gen_optim, **scheduler['kwargs'])
                dis_schedular = getattr(torch.optim.lr_scheduler, scheduler['type'])(dis_optim, **scheduler['kwargs'])
                return [gen_optim, dis_optim], [gen_schedular, dis_schedular]
            return [gen_optim, dis_optim]

        else:
            gen_optim = getattr(torch.optim, optimizer['type'])(self.gen.parameters(), **optimizer['kwargs'])
            if scheduler:
                gen_schedular = getattr(torch.optim.lr_scheduler, scheduler['type'])(gen_optim, **scheduler['kwargs'])
                return [gen_optim, gen_schedular]
            return gen_optim
