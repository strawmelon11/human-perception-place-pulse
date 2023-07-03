import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Net(nn.Module):
    def __init__(self, num_class = 2):
        super(Net, self).__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_fc = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_fc, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_class, bias=True)
        )
        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

def vit_place_pulse(perception = "lively", pretrained=True, **kwargs):
    model = Net(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://github.com/strawmelon11/human-perception-place-pulse/releases/download/v0.0.1/{perception}.pth")
        model.load_state_dict(state_dict)
    return model