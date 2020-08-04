# WaveFlow
**WaveFlow : A Compact Flow-based Model for Raw Audio**

WaveFlow, a smallfootprint generative flow for raw audio, which is directly trained with maximum likelihood. It handles the long-range structure of 1-D waveform with a dilated 2-D convolutional architecture, while modeling the local variations using expressive autoregressive functions. WaveFlow provides a unified view of likelihood-based models for 1-D data, including WaveNet and WaveGlow as special cases. It generates high-fidelity speech as WaveNet, while synthesizing several orders of magnitude faster as it only requires a few sequential steps to generate very long waveforms with hundreds of thousands of time-steps.


## For Preprocessing :

```
python .\nvidia_preprocessing.py -d path_of_wavs
```

## Training :

```
python .\train.py -c .\config\default.yaml -n "first"
```

## Inference :

```
python .\inference.py input_file output_file -r checkpoint_path --mel
```

