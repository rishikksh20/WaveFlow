import os
import argparse
import torch
from model.waveflow import WaveFlow
from utils.utils import remove_weight_norms
from librosa import load
from librosa.output import write_wav
from time import time
import math
import numpy as np
from utils.hparams import HParam, load_hparam_str
from utils.utils import set_deterministic_pytorch
from denoiser import Denoiser

def main(hp, checkpoint, infile, outfile, filename, sigma, dur, half, is_mel, device):
    set_deterministic_pytorch()
    # build model architecture
    model = model = WaveFlow(hp.model.flows,
                 hp.model.n_group,
                 hp.audio.sampling_rate,
                 hp.audio.win_length,
                 hp.audio.n_mel_channels,
                     hp).cuda()
    #model.summary()

    # load state dict
    state_dict = checkpoint['model']
    filename = filename + "_" +checkpoint['githash'] + "_" + str(checkpoint['epoch'])
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # if config['n_gpu'] > 1:
    #     model = model.module
    model.apply(remove_weight_norms)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    sr = hp.audio.sampling_rate
    if is_mel:
        mel = torch.from_numpy(np.load(infile)).unsqueeze(0).to(device)
        if half: 
            filename = filename + "_" + "fp16"
            mel = mel.half()
            model = model.half()
    else :
        y, _ = load(infile, sr=sr, duration=dur)
        
        offset = len(y) % hp.model.n_group 
        if offset:
            y = y[:-offset]

        y = torch.Tensor(y).to(device)

        # get mel before turn to half, because sparse.half is not implement yet
        mel = model.get_mel(y[None, :])

        if half:
            model = model.half()
            mel = mel.half()
            y = y.half()

        with torch.no_grad():
            start = time()
            z, logdet, _ = model(y[None, :], mel)
            cost = time() - start
            z = z.squeeze()
        
        print(z.mean().item(), z.std().item())
        print("Forward LL:", logdet.mean().item() / z.size(0) - 0.5 * (z.pow(2).mean().item() / sigma ** 2 + math.log(2 * math.pi) + 2 * math.log(sigma)))
        print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(cost, z.numel() / cost / 1000))
        
    
    start = time()
    x, logdet = model.infer(mel, sigma) # x -> [T]
    cost = time() - start
    audio = x
    if args.d:
        filename = filename + "_" + "d"
        denoiser = Denoiser(model, half=half, device=device).to(device)
        audio = denoiser(audio, 0.00015) # [B, 1, T]
        audio = audio.squeeze()
        audio = audio[:-(hp.audio.hop_length * 10)]

    print("Backward LL:", -logdet.mean().item() / x.size(0) - 0.5 * (1 + math.log(2 * math.pi) + 2 * math.log(sigma)))
    filename = filename + ".wav"
    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(cost, x.numel() / cost / 1000))
    print(x.max().item(), x.min().item())
    write_wav(os.path.join(outfile, filename), audio.cpu().float().numpy(), sr, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WaveGlow inference')
    parser.add_argument('infile', type=str, help='wave file to generate mel-spectrogram')
    parser.add_argument('--out', type=str,  default=".", help='output file name')
    parser.add_argument('--duration', type=float, help='duration of audio, in seconds')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--mel', action='store_true')
    parser.add_argument('-s', '--sigma', type=float, default=1.0)
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--chkpt', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', action='store_true', help="denoising ")
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    torch.manual_seed(2020)
    
    checkpoint = torch.load(args.chkpt)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])
    filename = "out"
    if args.cpu:
        filename = filename + "_" + "cpu"
        device = torch.device('cpu')
    else:
        filename = filename + "_" + "cuda"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Mel file input : ", args.mel)
    main(hp, checkpoint, args.infile, args.out, filename, args.sigma, args.duration, args.half, args.mel, device)
