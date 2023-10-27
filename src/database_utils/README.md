# Extracting Databases

## RealEstate-10K

> [!NOTE]
> The images downloaded by the below steps are slightly different from the images we downloaded a couple of years back. So, evaluating the images rendered by the pre-trained models with the images downloaded using below steps gives poor scores. We are looking into the discrepancy. In the meantime, if you need to obtain the QA scores of our model on the RE10K dataset for the specified scenes, please re-train SimpleNeRF on the downloaded images.

1. Download the dataset metadata from [here](https://google.github.io/realestate10k/download.html) and place it in `data/databases/RealEstate10K/downloaded_data/RealEstate10K.tar.gz`

2. Unzip the downloaded file
   ```shell
   cd data/databases/RealEstate10K
   mkdir unzipped_data
   tar -xzvf downloaded_data/RealEstate10K.tar.gz -C unzipped_data/
   cd ../../../
   ```

3. Obtain camera data of the five scenes used in ViP-NeRF
   ```shell
   cd src/database_utils/real_estate/data_organizers
   python VideoNameMapper.py
   ```

4. Run the data extractor file. This requires [youtube-dl](https://github.com/ytdl-org/youtube-dl) and [ffmpeg](https://ffmpeg.org/download.html) to be installed.
   ```shell
   python DataExtractor01.py
   cd ..
   ```
   If youtube-dl is not able to extract uploader-id, reinstall youtube-dl as suggested [here](https://stackoverflow.com/a/76409717/3337089).

5. train/test configs are already provided in the repository. In case you want to create them again:
   ```shell
   cd train_test_creators/
   python TrainTestCreator01.py
   python VideoPoseCreator01_Original.py
   cd ..
   ```

6. Return to root directory
```shell
cd ../../../
```

## NeRF-LLFF
1. Download the [`nerf_llff_data.zip`](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=share_link) file from original release in google drive. Place the downloaded file at `Data/databases/NeRF_LLFF/data/all/nerf_llff_data.zip`.

2. Run the data extractor file:
   ```shell
   cd src/database_utils/nerf_llff/data_organizers/
   python DataExtractor01.py
   cd ..
   ```

3. train/test configs are already provided in the repository. In case you want to create them again: 
   ```shell
   cd train_test_creators/
   python TrainTestCreator01_UniformSparseSampling.py
   python VideoPoseCreator01_Spiral.py
   cd ..
   ```

4. Return to root directory
   ```shell
   cd ../../../
   ```

## Custom Databases
We use the Open CV convention: `(x, -y, -z)` world-to-camera format to store the camera poses. 
The camera intrinsics and extrinsics are stored in the `csv` format after flattening them, i.e., if a scene contains 50 frames, intrinsics and extrinsics are stores as csv files with 50 rows each and 9 & 16 columns respectively.
The directory tree in the following shows an example.
Please refer to one of the [data-loaders](../data_loaders/RealEstateDataLoader01.py) for more details. 
Organize your custom dataset in accordance with the data-loader or write the data-loader file to load the data directly from your custom database format.

Example directory tree:
```shell
<DATABASE_NAME>
 |--data
    |--all
    |  |--database_data
    |     |--scene0001
    |     |  |--rgb
    |     |  |  |--0000.png
    |     |  |  |--0001.png
    |     |  |  |-- ...
    |     |  |  |--0049.png
    |     |  |--CameraExtrinsics.csv
    |     |  |--CameraIntrinsics.csv
    |     |--scene0002
    |     | ...
    |--train_test_sets
```

Our code also requires a config file specify the train/validation/test images. Please look into [train-test-creators](real_estate/train_test_creators/TrainTestCreator01.py) and replicate a similar file for your custom dataset.
