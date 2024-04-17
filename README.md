# 👏 Sclera-TransFuse  🔥

This repository contains the official code of Sclera-TransFuse: Fusing Vision Transformer and CNN for Accurate Sclera Segmentation and
Recognition

## Requirements
* `Python>=3.8`
* `Pytorch>=1.13.0` 
* `timm>=0.5`

## Ubiris.v2

## Experiments
1. **Dataset**
	+ Downloading training dataset and move it into `./data`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/17sUo2dLcwgPdO_fD4ySiS_4BVzc3wvwA/view?usp=sharing).
	+ Downloading testing dataset and move it into `./data` , which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1us5iOMWVh_4LAiACM-LQa73t1pLLPJ7l/view?usp=sharing).
2. **Testing**
	+ Downloading our trained DS-TransUNet-B from [Baidu Pan](https://pan.baidu.com/s/1EFZOX1C84mg1mVK6cAvpxg) (dd79), and move it into `./checkpoints`.
	+ run `test_kvasir.py`
	+ run `crcriteria.py` to get the DICE score, which uses [EvaluateSegmentation](https://github.com/Visceral-Project/EvaluateSegmentation). Or you can download our result images from [Baidu Pan](https://pan.baidu.com/s/1EFZOX1C84mg1mVK6cAvpxg) (dd79).
3. **Training**
	+ downloading `Swin-T` and `Swin-B` from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) to `./checkpoints`.
	+ run `train_kvasir.py`


Code of other tasks will be comming soon.


## Reference
Some of the codes in this repo are borrowed from:
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [PraNet](https://github.com/DengPingFan/PraNet)
* [TransFuse](https://github.com/Rayicer/TransFuse)



## Questions
Please drop an email to tianbaoge24@gmail.com



### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{HaiqingIJCB2023,
        title={Sclera-TransFuse: Fusing Swin Transformer and CNN for Accurate Sclera Segmentation},
        author={Li, Haiqing and Wang, Caiyong and Zhao, Guangzhe and He, Zhaofeng and Wang, Yunlong and Sun, Zhenan},
        booktitle={Proceedings of the IEEE International Joint Conference on Biometrics (IJCB)},
        pages={1--8},
        year={2023},
        organization={IEEE}

