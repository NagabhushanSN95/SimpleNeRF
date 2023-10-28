# SimpleNeRF
Official code release accompanying the SIGGRPH Asia 2023 paper - "SimpleNeRF: Regularizing Sparse Input Neural Radiance Fields with Simpler Solutions".

* [Project Page](https://nagabhushansn95.github.io/publications/2023/SimpleNeRF.html)
* [Published Data (OneDrive)](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/EgLoheuOaDdBg9V1wX5QCTYBmSg-QOPXGxZ6wYEH6y6cjA?e=t1iLUY)

## Python Environment
Environment details are available in `EnvironmentData/SimpleNeRF.yml`. The environment can be created using conda
```shell
conda env create -f SimpleNeRF.yml
```

## Add the source directory to PYTHONPATH
```shell
export PYTHONPATH=<ABSOLUTE_PATH_TO_SIMPLENERF_DIR>/src:$PYTHONPATH
```

## Set-up Databases
Please follow the instructions in [database_utils/README.md](src/database_utils/README.md) file to set up various databases. Instructions for custom databases are also included here.

## Generate Priors
### Sparse Depth Prior
Please follow the instructions in [prior_generators/sparse_depth/README.md](src/prior_generators/sparse_depth/README.md) file to generate sparse depth prior.

## Training and Inference
The files `RealEstateTrainerTester01.py`, `NerfLlffTrainerTester01.py` contain the code for training, testing and quality assessment along with the configs for the respective databases.
```shell
cd src/
python RealEstateTrainerTester01.py
python NerfLlffTrainerTester01.py
cd ../
```

## Inference with Pre-trained Models
The train configs are also provided in `runs/training/train****` folders for each of the scenes. Please download the trained models from `runs/training` directory in the published data (link available at the top) and place them in the appropriate folders. Disable the train call in the [TrainerTester](src/RealEstateTrainerTester01.py#L457) files and run the respective files. This will run inference using the pre-trained models and also evaluate the synthesized images and reports the performance.

## Evaluation
Evaluation of the rendered images will be automatically done after rendering the images. However, to compute depth based metrics and masked metrics, ground truth depth maps are needed. We obtain (pseudo) ground truth depth maps by training the vanilla NeRF with dense input views. Download these depth maps and visibility masks (for masked metrics) from `data` directory in the published data (link available at the top) and place them in the appropriate folders.

## License
MIT License

Copyright (c) 2023 Nagabhushan Somraj, Adithyan Karanayil, Rajiv Soundararajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Citation
If you use this code for your research, please cite our paper

```bibtex
@article{somraj2023SimpleNeRF,
    title = {{SimpleNeRF}: Regularizing Sparse Input Neural Radiance Fields with Simpler Solutions},
    author = {Somraj, Nagabhushan and Karanayil, Adithyan and Soundararajan, Rajiv},
    booktitle = {SIGGRAPH Asia},
    month = {December},
    year = {2023},
    doi = {10.1145/3610548.3618188},
}
```
If you use outputs/results of SimpleNeRF model in your publication, please specify the version as well. The current version is 1.0.

## Acknowledgements
Our code is built on top of [ViP-NeRF](https://github.com/NagabhushanSN95/ViP-NeRF) codebase.


For any queries or bugs regarding SimpleNeRF, please raise an issue.
