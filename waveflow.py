import torch
import torch.nn as nn
import numpy as np
from util import add_weight_norms
from modules import WN2D, InvertibleConv1x1
import torch.nn.functional as F
from librosa.filters import mel



class WaveFlowLoss(nn.Module):
    def __init__(self, sigma=1., elementwise_mean=True):
        super().__init__()
        self.sigma2 = sigma ** 2
        self.mean = elementwise_mean

    def forward(self, z, logdet):
        loss = 0.5 * z.pow(2).sum(1) / self.sigma2 - logdet
        loss = loss.mean()
        if self.mean:
            loss = loss / z.size(1)
        return loss

class WaveFlow(nn.Module):
    def __init__(self,
                 flows,
                 n_group,
                 sr,
                 window_size,
                 n_mels,
                 hp,
                 use_conv1x1 = False):
        super().__init__()
        self.flows = flows
        self.n_group = n_group
        self.win_size = window_size
        self.hop_size = hp.audio.hop_length
        self.n_mels = n_mels
        self.sr = sr
        self.sub_sr = self.hop_size // n_group

        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(n_mels, n_mels, self.sub_sr * 2 + 1, self.sub_sr, padding=self.sub_sr),
            nn.LeakyReLU(0.4, True)
        )
        self.upsampler.apply(add_weight_norms)

        self.WNs = nn.ModuleList()

        if use_conv1x1:
            self.invconv1x1 = nn.ModuleList()

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        for k in range(flows):
            self.WNs.append(WN2D(n_group, n_mels, hp.model.dilation_channels, hp.model.residual_channels,
                                 hp.model.skip_channels))
            if use_conv1x1:
                self.invconv1x1.append(InvertibleConv1x1(n_group, memory_efficient=False))

        filters = mel(sr, window_size, n_mels, fmax=8000)
        self.filter_idx = np.nonzero(filters)
        self.register_buffer('filter_value', torch.Tensor(filters[self.filter_idx]))
        self.filter_size = torch.Size(filters.shape)
        self.register_buffer('window', torch.hann_window(window_size))

    def get_mel(self, x):
        batch_size = x.size(0)
        S = torch.stft(x, self.win_size, self.hop_size, window=self.window, pad_mode='constant').pow(2).sum(3)
        mel_filt = torch.sparse_coo_tensor(self.filter_idx, self.filter_value, self.filter_size)
        N = S.size(1)
        mel_S = mel_filt @ S.transpose(0, 1).contiguous().view(N, -1)
        # compress
        mel_S.add_(1e-7).log_()
        return mel_S.view(self.n_mels, batch_size, -1).transpose(0, 1)

    def forward(self, x, h=None):
        if h is None:
            h = self.get_mel(x)
        y = self._upsample_h(h)

        batch_dim, n_mels, times = y.shape
        x = x.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y[..., :x.size(-1)]

        if hasattr(self, 'invconv1x1'):
            invconv1x1 = self.invconv1x1
        else:
            invconv1x1 = [None] * self.flows

        for k, (WN, invconv) in enumerate(zip(self.WNs, invconv1x1)):
            x0 = x[:, :, :1]
            log_s, t = WN(x[:, :, :-1], y)
            xout = x[:, :, 1:] * log_s.exp() + t

            if k:
                logdet += log_s.sum((1, 2, 3))
            else:
                logdet = log_s.sum((1, 2, 3))

            if invconv is None:
                x = torch.cat((xout.flip(2), x0), 2)
            else:
                x, log_det_W = invconv(torch.cat((x0, xout), 2).squeeze(1))
                x = x.unsqueeze(1)
                logdet += log_det_W

        return x.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet, h

    def _upsample_h(self, h):
        h = F.pad(h, (0, 1))
        # return F.interpolate(h, size=((h.size(2) - 1) * self.upsample_factor + 1,), mode='linear')
        return self.upsampler(h)

    def inverse(self, z, h):
        y = self._upsample_h(h)

        batch_dim, n_mels, times = y.shape
        z = z.view(batch_dim, 1, -1, self.n_group).transpose(2, 3).contiguous()
        y = y[..., :z.size(-1)]

        if hasattr(self, 'invconv1x1'):
            invconv1x1 = self.invconv1x1
        else:
            invconv1x1 = [None] * self.flows

        logdet = None
        for k, WN, invconv in zip(range(self.flows - 1, -1, -1), self.WNs[::-1], invconv1x1[::-1]):
            if invconv is None:
                z = z.flip(2)
            else:
                z, log_det_W = invconv.inverse(z.squeeze(1))
                z = z.unsqueeze(1)
                if logdet is None:
                    logdet = log_det_W.repeat(z.shape[0])
                else:
                    logdet += log_det_W

            xnew = z[:, :, :1]
            x = [xnew]

            buffer_list = None
            for i in range(1, self.n_group):
                log_s, t, buffer_list = WN.inverse_forward(xnew, y, buffer_list)
                xnew = (z[:, :, i:i + 1] - t) / log_s.exp()
                x.append(xnew)

                if logdet is None:
                    logdet = log_s.sum((1, 2, 3))
                else:
                    logdet += log_s.sum((1, 2, 3))
            z = torch.cat(x, 2)

        z = z.squeeze(1).transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, -logdet

    @torch.no_grad()
    def infer(self, h, sigma=1.):
        if h.dim() == 2:
            h = h[None, ...]

        batch_dim, n_mels, steps = h.shape
        samples = steps * self.hop_size

        z = h.new_empty((batch_dim, samples)).normal_(std=sigma)
        x, _ = self.inverse(z, h)
        return x.squeeze(), _