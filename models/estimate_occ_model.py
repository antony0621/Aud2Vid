"""
This is a simple version of Audio2Video implementation.
When training, image, which is used to provide appearance, is fed into the VAE-like image branch, and audio is fed into
an valina encoder. The combined code is the overall representation of the semantic, which will be decoded by a decoder
to generate corresponding video. In the test phase, the image branch is vectors from a Gaussian distribution.

The synthesized video contains the appearance of the input image, and its dynamic
property is defined by the audio. For instance, if the given audio is bird's singing, and the image is a description of
a bird in cage, then the generated video should be that bird is singing in a cage; however, if the given image is a bird
on a tree in a park, then the corresponding video should change accordingly.

And after this work, something related but maybe more interesting should also be tested. For example, generating video
through a piece of music. The model should be able to capture the content, i.e., what story is the piece sing about?
"""

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable as Vb

from src.utils import ops
from models.basemodule import ConvBNRelU, ConvBase, ConvBlock, UpConv, Conv3d
import src.opts as opts
from models.vgg.vgg_128 import Flow2Frame_warped
from models.vgg.vgg_utils import my_vgg


class AudioFrameNet(nn.Module):
    def __init__(self, z_dim=1024):
        """
        Obtain audio-frame combined representation x=(mu, logvar)
        When training, minimize the KL div between z here and z from VideoNet
        to let audio and few visual content describe as completed visual content as possible
        :param z_dim: the dimension of z
        """
        super(AudioFrameNet, self).__init__()

        self.audio_conv = nn.Sequential(ConvBNRelU(1, 96),
                                        ConvBNRelU(96, 256),
                                        ConvBNRelU(256, 512),
                                        ConvBNRelU(512, 512),
                                        ConvBNRelU(512, 512))
        self.audio_fc = nn.Linear(512 * 16 * 2, z_dim // 2)

        self.frame_conv1 = ConvBase(3, 32, 4, 2, 1)  # 32,64,64
        self.frame_conv2 = ConvBlock(32, 64, 4, 2, 1)  # 64,32,32
        self.frame_conv3 = ConvBlock(64, 128, 4, 2, 1)  # 128,16,16
        self.frame_conv4 = ConvBlock(128, 256, 4, 2, 1)  # 256,8,8
        self.frame_fc = nn.Linear(256 * 8 * 8, z_dim // 2)

        self.mu_fc = nn.Linear(z_dim, z_dim)
        self.logvar_fc = nn.Linear(z_dim, z_dim)

    def forward(self, audio, frame):
        """

        :param audio: [512, 90]
        :param frame: [3, 128, 128]
        :return:
        """
        x_a = self.audio_conv(torch.unsqueeze(audio, 1))
        x_a = x_a.view(-1, 512 * 16 * 2)
        x_a = self.audio_fc(x_a)

        x_f1 = self.frame_conv1(frame)
        x_f2 = self.frame_conv2(x_f1)
        x_f3 = self.frame_conv3(x_f2)
        x_f4 = self.frame_conv4(x_f3)  # 256, 8, 8
        x_f = x_f4.view(-1, 256 * 8 * 8)
        x_f = self.frame_fc(x_f)

        z = torch.cat((x_a, x_f), 1)
        mu = self.mu_fc(z)
        logvar = self.logvar_fc(z)
        return mu, logvar, [x_f1, x_f2, x_f3, x_f4]


class VideoNet(nn.Module):

    def __init__(self, input_channel, output_channel=1024):
        super(VideoNet, self).__init__()
        """
        Seg2Vid structure: direct input or residual input? 
        Take full-length frames as input, output Z_v
        Video: [N, time, 3, 128, 128]
        Z_v: 1024
        """
        # input 3*128*128
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, 2, 1, bias=False),  # 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),  # 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 4, 2, 1, bias=False)  # 8
        )
        self.fc1 = nn.Linear(64 * 8 * 8, output_channel)
        self.fc2 = nn.Linear(64 * 8 * 8, output_channel)

    def forward(self, x):
        temp = self.main(x)
        temp = temp.view(-1, 64 * 8 * 8)
        mu = self.fc1(temp)
        logvar = self.fc2(temp)
        return mu, logvar


class FlowNet(nn.Module):
    where = ["flow", "mask"]  # flow and mask is obtained after this block

    def __init__(self, opt):
        super(FlowNet, self).__init__()
        self.opt = opt
        self.dconv1 = ConvBlock(256 + 16, 256, 3, 1, 1)  # 256,8,8
        self.dconv2 = UpConv(256, 128, 3, 1, 1)  # 128,16,16
        self.dconv3 = UpConv(256, 64, 3, 1, 1)  # 64,32,32
        self.dconv4 = UpConv(128, 32, 3, 1, 1)  # 32,64,64
        self.gateconv1 = Conv3d(64, 64, 3, 1, 1)
        self.gateconv2 = Conv3d(32, 32, 3, 1, 1)

    def forward(self, enco1, enco2, enco3, z):
        opt = self.opt
        deco1 = self.dconv1(z)  # .view(-1,256,4,4,4)# bs*4,256,8,8
        deco2 = torch.cat(torch.chunk(self.dconv2(deco1).unsqueeze(2), opt.num_predicted_frames, 0),
                          2)  # bs*4,128,16,16
        deco2 = torch.cat(
            torch.unbind(torch.cat([deco2, torch.unsqueeze(enco3, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1),
                         2), 0)
        deco3 = torch.cat(self.dconv3(deco2).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # 128,32,32
        deco3 = self.gateconv1(deco3)
        deco3 = torch.cat(
            torch.unbind(torch.cat([deco3, torch.unsqueeze(enco2, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1),
                         2), 0)
        deco4 = torch.cat(self.dconv4(deco3).unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)  # 32,4,64,64
        deco4 = self.gateconv2(deco4)
        deco4 = torch.cat(
            torch.unbind(torch.cat([deco4, torch.unsqueeze(enco1, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)], 1),
                         2), 0)
        return deco4


class FlowPredictor(nn.Module):
    def __init__(self):
        super(FlowPredictor, self).__init__()
        self.main = nn.Sequential(
            UpConv(64, 16, 5, 1, 2),
            nn.Conv2d(16, 2, 5, 1, 2),
        )

    def forward(self, x):
        return self.main(x)


class MaskPredictor(nn.Module):
    def __init__(self):
        super(MaskPredictor, self).__init__()
        self.main = nn.Sequential(
            UpConv(64, 16, 5, 1, 2),
            nn.Conv2d(16, 2, 5, 1, 2),
        )

    def forward(self, x):
        return torch.sigmoid(self.main(x))


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.opt = opt
        # encoders
        self.aud_fra_enc = AudioFrameNet()
        self.video_enc = VideoNet(input_channel=(opt.num_predicted_frames + 1) * 3)
        # decoders
        self.z_conv = ConvBase(256 + 16, 16 * opt.num_predicted_frames, 3, 1, 1)  # for dim adaptation
        self.flow_net = FlowNet(opt)
        self.flow_dec = FlowPredictor()
        self.mask_dec = MaskPredictor()

        # post-processing module
        self.flow_wrapper = ops.FlowWrapper()
        self.refine_net = Flow2Frame_warped(num_channels=opt.input_channel)

        # load vgg for perceptual loss
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg_net = my_vgg(vgg19)
        for param in self.vgg_net.parameters():
            param.requires_grad = False

        pass

    def forward(self, audio, video=None):
        """
        When training, video is taken as input to estimate a posterior and to help the model to learn a prior from
        the audio; when test, the prior is directly calculated from the audio, from which the z is sampled.
        :param audio: temporal-frequent representation of the raw waveform
        :param video: full-length farmes of the video; video = None means test stage.
        :return: Combined representation of (audio, frame) and video representation
        """
        opt = self.opt

        frame = video[:, 0]  # (bs, 3, 128, 128)
        frames = video[:, 1:]  # (bs, t-1, 3, 128, 128)
        video = torch.cat([frame, frames.contiguous().view(-1, opt.num_predicted_frames * 3, 128, 128) -
                           frame.repeat(1, opt.num_predicted_frames, 1, 1)], 1)  # (bs, t*3, 128, 128)

        # obtain statistical results from the two distributions
        mu_af, logvar_af, hiddens = self.aud_fra_enc(audio, frame)  # learned prior, (bs, 1024), to sample from
        mu_v, logvar_v = self.video_enc(video)  # posterior, (bs, 1024), to train and learn the prior

        # sample from the prior distribution
        z_af = self.reparameterize(mu_af, logvar_af)  # (bs, 1024)

        # decode and obtain the flows and masks
        codex = hiddens[3]  # representation of the initial frame, (bs, 256, 8, 8)
        codey = torch.cat([z_af.view(-1, 16, opt.input_size[0] // 16, opt.input_size[1] // 16), codex], 1)  # check 16
        codey = self.z_conv(codey)  # (bs, 16 * n_pred_frames, 8, 8)
        codex = torch.unsqueeze(codex, 2).repeat(1, 1, opt.num_predicted_frames, 1, 1)  # (bs, 256, n_pred_frames, 8, 8)
        codey = torch.cat(torch.chunk(codey.unsqueeze(2), opt.num_predicted_frames, 1), 2)  # (..., 16, ...)
        z = torch.cat(torch.unbind(torch.cat([codex, codey], 1), 2), 0)  # (bs * n_pred_frames, 272, 8, 8)
        pre_dec = self.flow_net(hiddens[0], hiddens[1], hiddens[2], z)

        # flow computation
        flow_forward = self.flow_dec(pre_dec)  # (bs * n_pred_frames, 2, 128, 128), 2 means horizon and vertical
        flow_forward = torch.cat(flow_forward.unsqueeze(2).chunk(opt.num_predicted_frames, 0),
                                 2)  # (bs, 2, n_pred_frames, 128, 128)
        flow_backward = self.flow_dec(pre_dec)  # (bs * n_pred_frames, 2, 128, 128), 2 means horizon and vertical
        flow_backward = torch.cat(flow_backward.unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)
        # mask computation
        pred_mask = self.mask_dec(pre_dec)
        pred_mask = torch.cat(pred_mask.unsqueeze(2).chunk(opt.num_predicted_frames, 0), 2)
        mask_forward = pred_mask[:, 0, ...]
        mask_backward = pred_mask[:, 1, ...]

        # post-processing to obtain final frames
        warppede_fw = mask_fw * self.flow_wrapper(frame, flow_forward)
        pred = self.refine_net(warppede_fw, flow_forward)

        if video is not None:  # i.e. training
            prediction_vgg_feature = self.vgg_net(
                self._normalize(pred.contiguous().view(-1, opt.input_channel, opt.input_size[0], opt.input_size[1])))
            gt_vgg_feature = self.vgg_net(
                self._normalize(video.contiguous().view(-1, opt.input_channel, opt.input_size[0], opt.input_size[1])))

            return (mu_af, logvar_af, mu_v, logvar_v, flow_forward, flow_backward, mask_forward, mask_backward, pred,
                    prediction_vgg_feature, gt_vgg_feature)
        else:
            return flow_forward, flow_backward, mask_forward, mask_backward, pred

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Vb(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return Vb(mu.data.new(mu.size()).normal_())

    def _normalize(self, x):
        gpu_id = x.get_device()
        return (x - mean.cuda(gpu_id)) / std.cuda(gpu_id)


# the mean and the std of the MUSIC21 should be verified
mean = Vb(torch.FloatTensor([0.485, 0.456, 0.406])).view([1, 3, 1, 1])
std = Vb(torch.FloatTensor([0.229, 0.224, 0.225])).view([1, 3, 1, 1])

if __name__ == '__main__':
    # # audio and frame
    aud = torch.randn(1, 512, 90)
    # frame = torch.randn(1, 3, 128, 128)
    # AFN = AudioFrameNet()
    # x, y, z = AFN(aud, frame)
    # print(x.size())
    # print(y.size())
    # for z_cap in z:
    #     print(z_cap.size())
    #
    # # video
    vid = torch.randn(1, 5, 3, 128, 128)
    # VN = VideoNet(input_channel=3)
    # z_i, z_j = VN(vid)
    # print(z_i.size(), z_j.size())
    opt = opts.parse_opts()
    net = VAE(opt)
    mu, logvar, flow_fw, flow_bw, mask_fw, mask_bw = net(aud, vid)
    print(mu.size(), logvar.size(), flow_fw.size(), flow_bw.size(), mask_fw.size(), mask_bw.size())
    print(mask_fw)
