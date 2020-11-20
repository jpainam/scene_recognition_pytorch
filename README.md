### Configuration

define the params in the configuration file located as config

### Training
```
python train.py --config config/MITIndoor67.yaml
```

### Testing
```
python test.py --config config/MITIndoor67.yaml
```


### Results
```
Dataset loaded!
Train set. Size 5360
Validation set. Size 1340
Train set number of scenes: 67
Validation set number of scenes: 67

Checkpoint loaded!
Testing batch: [9/84]   Loss 0.018 (avg: 0.016) Prec@1 1.000 (avg: 1.000)       Prec@2 1.000 (avg: 1.000)     Prec@5 1.000 (avg: 1.000)
Testing batch: [19/84]  Loss 0.029 (avg: 0.016) Prec@1 0.984 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)     Prec@5 1.000 (avg: 1.000)
Testing batch: [29/84]  Loss 0.015 (avg: 0.015) Prec@1 1.000 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)       Prec@5 1.000 (avg: 1.000)
Testing batch: [39/84]  Loss 0.012 (avg: 0.015) Prec@1 1.000 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)       Prec@5 1.000 (avg: 1.000)
Testing batch: [49/84]  Loss 0.023 (avg: 0.017) Prec@1 0.984 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)       Prec@5 1.000 (avg: 1.000)
Testing batch: [59/84]  Loss 0.012 (avg: 0.017) Prec@1 1.000 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)       Prec@5 1.000 (avg: 1.000)
Testing batch: [69/84]  Loss 0.026 (avg: 0.017) Prec@1 1.000 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)       Prec@5 1.000 (avg: 1.000)
Testing batch: [79/84]  Loss 0.014 (avg: 0.018) Prec@1 1.000 (avg: 0.998)       Prec@2 1.000 (avg: 1.000)       Prec@5 1.000 (avg: 1.000)
Elapsed time for <class 'set'> set evaluation 10.490 seconds

-----------------------------------------------------------------
Evaluation statistics:
Validation results: Loss 0.018, Prec@1 0.998, Prec@2 1.000, Prec@5 1.000, Mean Class Accuracy 119.113

```