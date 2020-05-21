# old_photo_pytorch

This is an implementation of this [paper(Bringing Old Photos Back to Life)](http://raywzy.com/Old_Photo/).

## Requirements

```
Python 3.7.7
Pytorch 1.5.0
torchvision 0.6.0
```

## preparation

Currently, the folder of the dataset should be organized as follows

```
./data
├── rimg
│   ├── 0001_in.png
│   └── ...
├── ximg
│   ├── 0001_in.png
│   └── ...
└── yimg
    ├── 0001_gt.png
    └── ...
```

## Notice

Since the dataset has not been published by the author, this implementation version is not perfect at present and will be updated later.