# Rotor Blade Icing Detection with generative AI
Supplementary material for the application paper "Domain-Invariant Icing Detection on Wind Turbine Rotor Blades with Generative AI for Deep Transfer Learning" (in submission).


A repository with code for predicting blade icing on images of turbine rotor blades using supervised (neural style transfer) and unsupervised (CycleGAN) techniques.


## Download and use of the repository:
To download this repository and its submodules use

    git clone --recurse-submodules https://github.com/malvela/WindTurbine-IceDetection_GenerativeAI.git

## Individual files and functionality:
This repository contains Python files for generalised icing prediction (domain-invariant - independent of the wind park the AI model has been trained on) on wind turbine rotor blades using a tiny computer.

- cyclegan_generativeai_icing.py : Used to train the CycleGAN model from scratch (or leverage the pre-trained Summer2Winter Yosemite model).

## Cite as:

If you are using this repository in your research, please cite it as:


Chatterjee J., Alvela Nieto M.T., Gelbhardt H., Dethlefs N., Ohlendorf J.-H., Greulich A., Thoben K.-D., "Domain-Invariant Icing Detection on Wind Turbine Rotor Blades with Generative AI for Deep Transfer Learning" (in submission)
## License:

This repo is based on the MIT License, which allows free use of the provided resources, subject to the original sources being credit/acknowledged appropriately. The software/resources under MIT license is provided as is, without any liability or warranty at the end of the authors.
