import sys
import torch
from utils.stft import STFT


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveflow, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros', half=False, device = torch.device('cuda')):
        super(Denoiser, self).__init__()
        self.device = device
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length/n_overlap),
                         win_length=win_length).cuda()
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88)).to(device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88)).to(device)
        else:
            raise Exception("Mode {} if not supported".format(mode))
        if half:
            mel_input = mel_input.half()
        with torch.no_grad():
            bias_audio, _ = waveflow.infer(mel_input) # [B, 1, T]
            bias_spec, _ = self.stft.transform(bias_audio.unsqueeze(0).float())

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.unsqueeze(0).to(self.device).float())
        audio_spec_denoised = audio_spec.to(self.device) - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles.to(self.device))
        print(audio_denoised.shape)
        return audio_denoised