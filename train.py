import torch
import os
import time
import logging
import argparse
import tqdm
from utils.hparams import HParam
from utils.utils import get_commit_hash, num_params
from torch.utils.data import DataLoader
from model.waveflow import WaveFlow, WaveFlowLoss
from dataset.melgan_dataloader import create_dataloader
from tensorboardX import SummaryWriter
import itertools
from utils.stft import TacotronSTFT

def train(args, chkpt_dir, chkpt_path, writer, logger, hp, hp_str, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    criterion = WaveFlowLoss(hp.model.sigma)
    model = WaveFlow(hp.model.flows,
                 hp.model.n_group,
                 hp.audio.sampling_rate,
                 hp.audio.win_length,
                 hp.audio.n_mel_channels,
                     hp).cuda()

    num_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam.lr)


    # Load checkpoint if one exists

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        #logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # trainset = Mel2Samp(hp.data.train, hp.audio.segment_length, hp.audio.filter_length,
    #              hp.audio.hop_length, hp.audio.win_length, hp.audio.sampling_rate,
    #                     hp.audio.mel_fmin, hp.audio.mel_fmax)
    #
    # validset = Mel2Samp(hp.data.valid, hp.audio.segment_length, hp.audio.filter_length,
    #              hp.audio.hop_length, hp.audio.win_length, hp.audio.sampling_rate,
    #                     hp.audio.mel_fmin, hp.audio.mel_fmax)



    # =====START: ADDED FOR DISTRIBUTED======
    # train_sampler = None
    # =====END:   ADDED FOR DISTRIBUTED======
    # train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
    #                           sampler=train_sampler,
    #                           batch_size=hp.train.batch_size,
    #                           pin_memory=False,
    #                           drop_last=True)
    #
    # valid_loader = DataLoader(validset, num_workers=1, shuffle=False,
    #                           sampler=train_sampler,
    #                           batch_size=1,
    #                           pin_memory=False,
    #                           drop_last=True)
    train_loader = create_dataloader(hp, True)
    valid_loader = create_dataloader(hp, False)
    # Get shared output_directory ready
    stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                        hop_length=hp.audio.hop_length,
                        win_length=hp.audio.win_length,
                        n_mel_channels=hp.audio.n_mel_channels,
                        sampling_rate=hp.audio.sampling_rate,
                        mel_fmin=hp.audio.mel_fmin,
                        mel_fmax=hp.audio.mel_fmax)

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in itertools.count(init_epoch + 1):
        if epoch % hp.log.validation_interval == 0:
            with torch.no_grad():
                pass

        loader = tqdm.tqdm(train_loader, desc='Loading train data')
        loss_list = []
        for (mel, audio) in loader:
            model.zero_grad()
            #mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda()) # [B, num mel, num of frame]
            audio = torch.autograd.Variable(audio.cuda()) # [B, T]
            z, logdet, _ = model(audio, mel) # [B, T]
            loss = criterion(z, logdet)

            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

            loader.set_description("Avg Loss : loss %.04f | step %d" % (sum(loss_list) / len(loss_list), step))
            if step % hp.log.summary_interval == 0:
                writer.add_scalar("train.loss", loss.item(), step)
                writer.add_scalar('train.log_determinant', logdet.mean().item())
                writer.add_scalar('train.z_mean', z.mean().item())
                writer.add_scalar('train.z_std', z.std().item())

            if step % hp.log.validation_interval == 0:
                for (mel_, audio_) in valid_loader:
                    x, logdet_ = model.infer(mel_.cuda(), hp.model.sigma) # x -> [T]

                    torch.clamp(x, -1, 1, out=x)

                    writer.add_scalar('valid.log_determinant', logdet.mean().item())
                    writer.add_audio('actual_audio', audio_.squeeze(0).cpu().detach().numpy(), step, sample_rate=hp.audio.sampling_rate)
                    writer.add_audio('reconstruct_audio', x.cpu().detach().numpy(), step, sample_rate=hp.audio.sampling_rate)
                    mel_spec = mel_[0].cpu().detach()
                    mel_spec -= mel_spec.min()
                    mel_spec /= mel_spec.max()
                    writer.add_image('actual_mel-spectrum', mel_spec.flip(0), step, dataformats='HW')

                    # mel_gen, _ = stft.mel_spectrogram(x)
                    # mel_g_spec = mel_gen[0].cpu()
                    # mel_g_spec -= mel_g_spec.min()
                    # mel_g_spec /= mel_g_spec.max()
                    # writer.add_image('gen_mel-spectrum', mel_g_spec.flip(0), step, dataformats='HW')
                    break


            step += 1

        if epoch % hp.log.save_interval == 0:
            save_path = os.path.join(chkpt_dir, '%s_%s_%04d.pt'
                                     % (args.name, githash, epoch))
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
                'githash': githash,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = SummaryWriter(log_dir)

    assert hp.audio.hop_length == 256, \
        'hp.audio.hop_length must be equal to 256, got %d' % hp.audio.hop_length
    assert hp.data.train != '' and hp.data.validation != '', \
        'hp.data.train and hp.data.validation can\'t be empty: please fix %s' % args.config


    train(args, pt_dir, args.checkpoint_path, writer, logger, hp, hp_str, hp.train.seed)
