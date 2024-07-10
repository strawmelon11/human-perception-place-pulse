# coding=UTF-8
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from transformers import AutoModel
from huggingface_hub import snapshot_download

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

perception = ['safety', 'lively', 'wealthy',
              'beautiful', 'boring', 'depressing']
model_dict = {
    'safety': 'safety.pth',
    'lively': 'lively.pth',
    'wealthy': 'wealthy.pth',
    'beautiful': 'beautiful.pth',
    'boring': 'boring.pth',
    'depressing': 'depressing.pth',
}


train_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


def predict(model, img_path, device):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = train_transform(img)
    img = img.view(1, 3, 384, 384)
    # inference
    if device == 'cuda:0':
        pred = model(img.cuda())
    else:
        pred = model(img)
    softmax = nn.Softmax(dim=1)
    pred = softmax(pred)[0][1].item()
    pred = round(pred*10, 2)

    return pred


if __name__ == "__main__":
    
    model_load_path = "./model"   # model dir path
    images_path = "./test_image"      # input image path
    out_Path = "./output"     # output path
    
    # download model
    print("Downloading models ...")
    snapshot_download(repo_id="Jiani11/human-perception-place-pulse",
                      allow_patterns=["*.pth", "README.md"], local_dir=model_load_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using device:{} ".format(device))
    for p in perception:
        # load model
        model_path = model_load_path + "/" + model_dict[p]
        model = torch.load(model_path, map_location=torch.device(device))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        print("######### {}  #########".format(p))
        model.eval()
        
        # Check if the directory exists, if not, create it
        if not os.path.exists(out_Path):
            os.makedirs(out_Path)
    
        # inferring img
        out_csvPath = out_Path + "/" + str(p) + ".csv"
        df = pd.DataFrame(columns=['img_path', str(p)+"_score"])
        df.to_csv(out_csvPath, index=False)
        data_arr = []

        for img in os.listdir(images_path):
            img_path = images_path + "/" + img
            print("current image: ", img_path)
            score = predict(model, img_path, device)
            data_arr = [img, score]
            df = pd.DataFrame(data_arr).T
            df.to_csv(out_csvPath, mode='a', header=False,
                      index=False)  # save scores into csv
        print("{} prediction done!".format(p))
