import pandas as pd
from pathlib import Path
from skimage import io
import feature_A
import feature_B
import feature_C
import feature_D
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import numpy as np


def process_csv(metadata_file, annotations_file, output_file):
    metadata_df = pd.read_csv(metadata_file)
    annotations_df = pd.read_csv(annotations_file)
    df = metadata_df.merge(annotations_df, on='img_id', how='inner')

    def process_row(row):
        try:
            img_name = row['img_id']

            parts = img_name.split('_')
            id_value = f"{parts[0]}_{parts[1]}"

            masks_path = Path(__file__).resolve().parent.parent / "data" / "masks"
            img_path = Path(__file__).resolve().parent.parent / "data" / "imgs"
            mask_name = img_name[:-4] + "_mask" + img_name[-4:]
            mask_path = masks_path / mask_name

            image_path = img_path / img_name

            if not mask_path.exists() or not image_path.exists():
                print(f"Missing image: {img_name}")
                return None

            mask = io.imread(mask_path)
            img = io.imread(image_path)
            
            mask1 = resize(mask, img.shape[:2], order=0, preserve_range=True).astype(bool)
            mask2 = resize(mask, img.shape[:2], order=0, preserve_range=True).astype(np.uint8) 

            asymmetry = feature_A.get_asymmetry(mask1)
            compactness = feature_B.get_compactness(mask1)
            hue, saturation, brightness = feature_C.get_hsv_mean(img,mask1)
            phue, psaturation, pbrightness = feature_D.pigmentation(img,mask2)

            diagnosis = row['diagnostic'].strip().upper()

            cancer_labels = ['BCC', 'SCC', 'MEL']
            cancer_flag = "Cancerous" if diagnosis in cancer_labels else "Benign"

            hair_mean = pd.Series([row['hair_1'], row['hair_2'], row['hair_3'], row['hair_4'], row['hair_5']]).mean()

            return pd.Series([id_value, img_name, asymmetry, compactness, hue, saturation, brightness, phue, psaturation, pbrightness, hair_mean, diagnosis, cancer_flag])

        except Exception as e:
            print(f"Skipping {row['img_id']} due to error: {e}")
            return None

    result = df.apply(process_row, axis=1)
    result = result.dropna()
    result.columns = ['ID', 'Image_ID', 'A_asymmetry', 'B_compactness', 'C_hue', 'C_saturation', 'C_brightness', 'D_hue', 'D_saturation', 'D_brightness','Hair_mean', 'Diagnosis', 'Cancer']


    feature_cols = [
        'A_asymmetry',
        'B_compactness',
        'C_hue',
        'C_saturation',
        'C_brightness',
        'D_hue',
        'D_saturation',
        'D_brightness',
        'Hair_mean'
    ]

    scaler = StandardScaler()
    result[feature_cols] = scaler.fit_transform(result[feature_cols])

    result.to_csv(output_file, index=False)


metadata_csv = Path(__file__).resolve().parent.parent / "data" / "metadata.csv"
annotations_csv = Path(__file__).resolve().parent.parent / "data" / "annotations_combined.csv"
output_csv = metadata_csv.parent / "features.csv"

process_csv(metadata_csv, annotations_csv, output_csv)