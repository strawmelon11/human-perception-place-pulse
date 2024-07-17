# human-perception-place-pulse

## What does the model do
### safety, lively, beautiful, wealthy, boring and depressing.
Getting human perception scores from street-level imagery. 

The scores are in scale of 0-10.

` Safety, lively, beautiful, wealthy`  high score indicates strong **positive** feeling

` Boring, depressing`  high score indicates strong **negative** feeling

Model Accuracy：
| Model | safe | lively | wealthy | beautiful | boring | depressing |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Accuracy | 76.7% | 77.1% | 72.9% | 76.9% | 61.6% | 67.2% |


## Model
The models are pre-trained on the MIT Place Pulse 2.0 dataset. The backbone of the models are vision transformer (ViT) pretrianed on ImageNet (ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1). 

In the ViT heads, 3 Linear layers with ReLU activiation are added for classification.  

The models will be automatically downloaded when run *eval.py* (recommended method). You can also manually download the models [here](https://huggingface.co/Jiani11/human-perception-place-pulse)


## How to run the model
Install packages from requirements.txt

` pip install -r requirements.txt` 

Change the file path in *eval.py*

```
model_load_path = "./model"   # path to save downloaded models
images_path = "./test_image"      # input image path
out_Path = "./output"     # output scores path
```
Run the file *eval.py*

`python eval.py`

## Citation
Please cite our papers if you use this code or any of the models. Find more streetscapes [here](https://github.com/ualsg/global-streetscapes)
```
@article{2024_global_streetscapes,
 author = {Hou, Yujun and Quintana, Matias and Khomiakov, Maxim and Yap, Winston and Ouyang, Jiani and Ito, Koichi and Wang, Zeyu and Zhao, Tianhong and Biljecki, Filip},
 doi = {10.1016/j.isprsjprs.2024.06.023},
 journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
 pages = {216-238},
 title = {Global Streetscapes -- A comprehensive dataset of 10 million street-level images across 688 cities for urban science and analytics},
 volume = {215},
 year = {2024}
}
```

