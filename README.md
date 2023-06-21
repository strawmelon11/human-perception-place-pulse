# human-perception-place-pulse

## What does the model do
### safety, lively, beautiful, wealthy, boring and depressing.
Getting human perception scores from street-level imagery. 

The scores are in scale of 0-10.

` Safety, lively, beautiful, wealthy`  **high** score indicates strong positive feeling

` Boring, depressing`  **high** score indicates strong negative feeling

## Model
We used vision transformer model pretrianed on ImageNet (ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1) as base model, and add 3 Linear layers in ViT heads for classification.

## How to run the model
Install packages from requirements.txt

` pip install -r requirements.txt` 

Change the file path in *eval.py*

```
model_load_path = "D:/model/"   # model path
images_path = "D:/test_image"      # your input image path
out_Path = "D:/output"     # output path
```
Run the file *eval.py*

`python eval.py`

## References
