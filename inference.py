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

def main(hp, checkpoint, infile, outfile, sigma, dur, half, is_mel):
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
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # if config['n_gpu'] > 1:
    #     model = model.module
    model.apply(remove_weight_norms)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    sr = hp.audio.sampling_rate
    if is_mel:
        mel = torch.from_numpy(np.load(infile)).unsqueeze(0).to(device)   
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
    x, logdet = model.infer(mel, sigma)
    cost = time() - start

    print("Backward LL:", -logdet.mean().item() / x.size(0) - 0.5 * (1 + math.log(2 * math.pi) + 2 * math.log(sigma)))

    print("Time cost: {:.4f}, Speed: {:.4f} kHz".format(cost, x.numel() / cost / 1000))
    print(x.max().item(), x.min().item())
    write_wav(outfile, x.cpu().float().numpy(), sr, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WaveGlow inference')
    parser.add_argument('infile', type=str, help='wave file to generate mel-spectrogram')
    parser.add_argument('outfile', type=str, help='output file name')
    parser.add_argument('--duration', type=float, help='duration of audio, in seconds')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--mel', action='store_true')
    parser.add_argument('-s', '--sigma', type=float, default=1.0)
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--chkpt', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    checkpoint = torch.load(args.chkpt)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print("Mel file input : ", args.mel)
    main(hp, checkpoint, args.infile, args.outfile, args.sigma, args.duration, args.half, args.mel)
