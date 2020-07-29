# WaveFlow
WaveFlow : A Compact Flow-based Model for Raw Audio

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

