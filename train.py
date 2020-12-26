import torch
from torch.autograd import Variable as Vb
import torch.optim as optim
from torch.utils.data import DataLoader
import os, time, sys

from models.estimate_occ_model import *
from src.utils import utils, ops
import losses
import dataset as ds
from src.opts import parse_opts

opt = parse_opts()
print(opt)


class Aud2Vid(object):

    def __init__(self, opt):

        self.opt = opt
        dataset = 'MUSIC21'
        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        self.job_name = dataset + '_gpu8_estimate_mask_'
        self.model_dir = self.job_name + 'model'
        self.sample_dir = os.path.join(self.workspace, self.job_name)
        self.parameter_dir = self.sample_dir + '/params'

        if not os.path.exists(self.parameter_dir):
            os.makedirs(self.parameter_dir)

        # whether to start training from an existing snapshot
        self.load = False
        self.iter_to_load = 62000

        # Write parameters setting file
        if os.path.exists(self.parameter_dir):
            utils.save_parameters(self)

        '''MUSIC21'''
        self.trainloader, self.valloader = ds.get_dataloader(root=opt.root, tag_dir=opt.train_tag_json_path, is_training=True)
        self.testloader = ds.get_dataloader(root=opt.root, tag_dir=opt.test_tag_json_path, is_training=False)

    def train(self):

        opt = self.opt
        gpu_ids = range(torch.cuda.device_count())
        print('Number of GPUs in use {}'.format(gpu_ids))

        iteration = 0

        vae = VAE(opt=opt).cuda()
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae).cuda()

        objective_func = losses.LossesMaskEst(opt, vae.module.floww)

        print(self.job_name)

        optimizer = optim.Adam(vae.parameters(), lr=opt.lr_rate)

        if self.load:

            model_name = self.sample_dir + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
            print("loading model from {}".format(model_name))

            state_dict = torch.load(model_name)
            if torch.cuda.device_count() > 1:
                vae.module.load_state_dict(state_dict['vae'])
                optimizer.load_state_dict(state_dict['optimizer'])
            else:
                vae.load_state_dict(state_dict['vae'])
                optimizer.load_state_dict(state_dict['optimizer'])
            iteration = self.iter_to_load + 1

        for epoch in range(opt.num_epochs):

            print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
            print('-' * 10)

            for video, audio in iter(self.trainloader):

                # get the inputs
                video = video.cuda()
                audio = audio.cuda()

                frame = video[:, 0, :, :, :]
                frames = video[:, 1:, :, :, :]

                start = time.time()

                # Set train mode
                vae.train()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                mu_af, logvar_af, mu_v, logvar_v, flow_forward, flow_backward, mask_forward, mask_backward, pred,
                prediction_vgg_feature, gt_vgg_feature = vae(audio, video)

                # Compute losses
                flowloss, reconloss, reconloss_back, reconloss_before, kldloss, flowcon, sim_loss, vgg_loss, mask_loss = objective_func(
                    frame1, frame2,
                    y_pred, mu, logvar, flow, flowback,
                    mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,
                    y_pred_before_refine=y_pred_before_refine)

                loss = (
                            flowloss + 2. * reconloss + reconloss_back + reconloss_before + kldloss * self.opt.lamda + flowcon + sim_loss + vgg_loss + 0.1 * mask_loss)

                # backward
                loss.backward()

                # Update
                optimizer.step()
                end = time.time()

                # print statistics
                if iteration % 20 == 0:
                    print(
                        "iter {} (epoch {}), recon_loss = {:.6f}, recon_loss_back = {:.3f}, "
                        "recon_loss_before = {:.3f}, flow_loss = {:.6f}, flow_consist = {:.3f}, kl_loss = {:.6f}, "
                        "img_sim_loss= {:.3f}, vgg_loss= {:.3f}, mask_loss={:.3f}, time/batch = {:.3f}"
                            .format(iteration, epoch, reconloss.item(), reconloss_back.item(), reconloss_before.item(),
                                    flowloss.item(), flowcon.item(),
                                    kldloss.item(), sim_loss.item(), vgg_loss.item(), mask_loss.item(), end - start))

                if iteration % 500 == 0:
                    utils.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration,
                                       self.sample_dir, opt)

                if iteration % 2000 == 0:
                    # Set to evaluation mode (randomly sample z from the whole distribution)
                    with torch.no_grad():
                        vae.eval()
                        val_sample, val_bg_mask, val_fg_mask, _ = iter(self.testloader).next()

                        # Read data
                        data = val_sample.cuda()
                        bg_mask = val_bg_mask.cuda()
                        fg_mask = val_fg_mask.cuda()
                        frame1 = data[:, 0, :, :, :]

                        noise_bg = torch.randn(frame1.size()).cuda()
                        y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw = vae(frame1, data,
                                                                                                         bg_mask,
                                                                                                         fg_mask,
                                                                                                         noise_bg)

                    utils.save_samples(data, y_pred_before_refine, y_pred, flow, mask_fw, mask_bw, iteration,
                                       self.sample_dir, opt,
                                       eval=True, useMask=True)

                    # Save model's parameter
                    checkpoint_path = self.sample_dir + '/{:06d}_model.pth.tar'.format(iteration)
                    print("model saved to {}".format(checkpoint_path))

                    if torch.cuda.device_count() > 1:
                        torch.save({'vae': vae.state_dict(), 'optimizer': optimizer.state_dict()},
                                   checkpoint_path)
                    else:
                        torch.save({'vae': vae.module.state_dict(), 'optimizer': optimizer.state_dict()},
                                   checkpoint_path)

                iteration += 1


if __name__ == '__main__':
    a = Aud2Vid(opt)
    a.train()
