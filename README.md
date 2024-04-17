# 👏 Sclera-TransFuse  🔥

This repository contains the testing code of Sclera-TransFuse: Fusing Vision Transformer and CNN for Accurate Sclera Segmentation and
Recognition

## Requirements
* `Python>=3.8`
* `Pytorch>=1.13.0` 
* `timm>=0.5`

## Ubiris.v2
A UBIRIS.v2 subset of 683 eye images with manually labeled sclera masks, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/18eW1JGHnv1SRyoMzSEq_3mmlRJ0_KgZ6/view?usp=sharing) and [download link (Baidu Pan)](https://pan.baidu.com/s/1WJ6anKO3d9MR6vzrXkX6pQ?pwd=p879) . 
## Experiments
1. **Sclera-TransFuse-Seg**
   +  Downloading our trained weight from [Google Drive](https://drive.google.com/drive/folders/104TAazlhPCHCWI1IcPqgUJKLjj1alGpr?usp=drive_link), and move it into `./checkpoints`
	+ modify the path in `Sclera_TransFuse.py` 
	+ modify the path in `testing.py`
	+ run `testing.py`
2.  **How to training Sclera-TransFuse-Rec**
  	+ modify --train_root_path=" "   --train_list=" "  and --save_path=" " in "training.bash"
	+ run `training.bash`
3. **Feature extraction and matching**
	+ Downloading our trained weight from [Google Drive](https://drive.google.com/file/d/1ZvQPEork9z9z01KM376Lp5APxYS3hfW6/view?usp=drive_link)
	+ modify `matching.py`
     + run `matching.py`

These codes are not the final version.



## Reference
Some of the codes in this repo are borrowed from:
* [LightCNN](https://github.com/AlfredXiangWu/LightCNN)
* [DS-TransUNet](https://github.com/TianBaoGe/DS-TransUNet)
* [Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)


## Questions
Please drop an email to haiqing_li@stu.bucea.edu.cn



### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{HaiqingIJCB2023,
        title={Sclera-TransFuse: Fusing Swin Transformer and CNN for Accurate Sclera Segmentation},
        author={Li, Haiqing and Wang, Caiyong and Zhao, Guangzhe and He, Zhaofeng and Wang, Yunlong and Sun, Zhenan},
        booktitle={Proceedings of the IEEE International Joint Conference on Biometrics (IJCB)},
        pages={1--8},
        year={2023},
        organization={IEEE}

