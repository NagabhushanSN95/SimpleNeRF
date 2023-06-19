# Extracting Databases

## RealEstate-10K
1. Download the dataset metadata from [here](https://google.github.io/realestate10k/download.html)
2. Remap the names to integers:
```shell
python VideoNameMapper.py
```
3. We use the scenes from test set. Select the scenes to download:
```shell
python SceneSelector01.py
```
4. Download the selected scenes:
```shell
python DataExtractor01.py
```
5. Copy the downloaded scenes to `Data/databases/RealEstate10K/data/test/database_data`.
6. Create the train/test configs: 
```shell
python TrainTestCreator01.py
python VideoPoseCreator01_Original.py
```

## NeRF-LLFF
1. Download the [`nerf_llff_data.zip`](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=share_link) file from original release in google drive. Place the downloaded file at `Data/databases/NeRF_LLFF/data/nerf_llff_data.zip`.
2. Run the data extractor file:
```shell
python DataExtractor01.py
```
3. Create the train/test configs: 
```shell
python TrainTestCreator01_UniformSparseSampling.py
python VideoPoseCreator01_Spiral.py
```

## Custom Databases
We use the Open CV convention - `(x, -y, -z)` world-to-camera format to store the camera poses. The camera intrinsics and extrinsics are stored in the `csv` format. Please refer to one of the [data-loaders](../data_loaders/RealEstateDataLoader01.py) for more details. Organize your custom dataset in accordance with the data-loader or write the data-loader file to load the data directly from your custom database format.

Our code also requires a config file specify the train/validation/test images. Please look into [train-test-creators](real_estate_10k/train_test_creators/TrainTestCreator01.py) and replicate a similar file for your custom dataset.
