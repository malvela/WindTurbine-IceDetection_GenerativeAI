# Rotor Blade Icing Detection with Generative AI
Supplementary material for the application paper "Domain-Invariant Icing Detection on Wind Turbine Rotor Blades with Generative AI for Deep Transfer Learning" (in submission).

<img src="https://user-images.githubusercontent.com/68553692/176689481-5fc86870-d7ed-4ec5-bb98-7abaf9564dc4.png" width="400" height="300" />

A repository with code for predicting blade icing on images of turbine rotor blades using supervised (neural style transfer) and unsupervised (CycleGAN) techniques.


## Download and use of the repository:
To download this repository and its submodules use

    git clone --recurse-submodules https://github.com/malvela/WindTurbine-IceDetection_GenerativeAI.git

## Individual files and functionality:
This repository contains Python files for generalised icing prediction (domain-invariant - independent of the wind park the AI model has been trained on) on wind turbine rotor blades using a tiny computer.

- CyleGAN/cyclegan_generativeai_icing.py : Used to train the CycleGAN model from scratch (or leverage the pre-trained Summer2Winter Yosemite model).
- Fast_Style_Transfer/Overlay_Images.ipynb : Used to overlay the styled image to the rotorblade using labelled masks.
- Fast_Style_Transfer/StyleTransfer_Notebook_BladeImages.ipynb: Used to modify the content images with the pretrained style transfer model. 
- StyleTransfer-TrainFromScratch/NST_TrainingFromScratch.py: Used to modidfy content images by training neural style transfer algorithm (based on VGG-19) from scratch.
## Cite as:

If you are using this repository in your research, please cite it as:


Chatterjee J., Alvela Nieto M.T., Gelbhardt H., Dethlefs N., Ohlendorf J.-H., Greulich A., Thoben K.-D., "Domain-Invariant Icing Detection on Wind Turbine Rotor Blades with Generative AI for Deep Transfer Learning" (in submission)
## License:

This repo is based on the MIT License, which allows free use of the provided resources, subject to the original sources being credit/acknowledged appropriately. The software/resources under MIT license is provided as is, without any liability or warranty at the end of the authors.
