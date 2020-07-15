name ="PyTorch WaveFlow"
n_gpu = 1
type = "WaveFlow"
flows = 8
n_group = 64
sample_rate = 22050
window_size = 1024
n_mels = 80
use_conv1x1 = False
memory_efficient = False
dilation_channels = 64
residual_channels = 64
skip_channels = 64
bias = False

# data_loader
# "RandomWaveFileLoader",

data_dir = "/host/data_disk/speech_datasets/LJSpeech-1.1/wavs"
batch_size = 2
num_workers = 1
segment = 16000

#  optimizer
# "Adam"
lr = 2e-4
step_size = 800000000
gamma = 0.5
# loss
# "WaveGlowLoss"

sigma = 1.0,
elementwise_mean = True

# Training
steps = 3000000,
save_dir = "./checkpoints/",
save_freq = 10000,
verbosity = 2

log_dir = "./logs/"
