# Generate Adversarial Examples of Fingerprint

file "model" is pretrained myCNN model by GPU

## CNN model train
```
python train.py
```

## (when test and never train, please use computer with GPU. Because the pretrained model is trained by GPU)

## CNN model normal fingerprint image test
```
python test.py
```

## CNN model generator of random test
```
python test.py random
```

## CNN model generator of FGSM test
```
python test.py fgsm
```

## CNN model generator of Fooling Image test
```
python test.py fool
```

## common fingerprint recognition test
```
python match.py
```