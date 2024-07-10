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



## How to run the model
Install packages from requirements.txt

` pip install -r requirements.txt` 

Change the file path in *eval.py*

```
model_load_path = "./model"   # model path
images_path = "./test_image"      # input image path
out_Path = "./output"     # output path
```
Run the file *eval.py*

`python eval.py`

## References
