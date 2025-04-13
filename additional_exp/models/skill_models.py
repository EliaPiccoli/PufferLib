import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision import transforms

from collections import namedtuple

# TODO: Eventually can become: Skill(input_model, input_output, skill_model, skill_output, adapter_model, adapter_output)
Skill = namedtuple('Skill', ['name', 'input_adapter', 'skill_model', 'skill_output', 'skill_adapter'])

def model_forward(model, x):
    return model(x).float()

def state_rep_input_trans(x: Tensor):
    x = F.interpolate(x, size=(160, 210), mode='bilinear', align_corners=False)
    x = x.repeat(1,4,1,1)
    return x

def get_state_rep_uns(game, device, expert=False):
    input_transformation_function = state_rep_input_trans
    if expert:
        model_path = "skills/models/" + game.lower() + "-state-rep-expert.pt"
    else:
        model_path = "skills/models/" + game.lower() + "-state-rep.pt"

    # n = Namespace()
    # setattr(n, 'feature_size', 512)
    # setattr(n, 'no_downsample', True)
    # setattr(n, 'end_with_relu', False)
    model = NatureCNN(4, 512)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    model.eval()
    model.to(device)
    model = torch.jit.script(model)
    adapter = None

    return Skill("state_rep_uns", input_transformation_function, model, model_forward, adapter)


def get_swin(device="cuda:0"):
    model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT).to(device)
    # model.head = torch.nn.Identity()
    model.eval()
    return Skill("swin", None, model, model_forward, None)

def get_resnet(device="cuda:0"):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).to(device)
    # model.head = torch.nn.Identity()
    model.fc = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()
    model.eval()
    return Skill("resnet", None, model, model_forward, None)

def pad_to_clip(tensor: torch.Tensor, target_size: int = 224) -> torch.Tensor:
    """
    Pads a batch of image tensors with zeros to reach the target size.

    Args:
        tensor (torch.Tensor): Input of shape [B, C, H, W]
        target_size (int): Target height and width

    Returns:
        torch.Tensor: Zero-padded tensor of shape [B, C, target_size, target_size]
    """
    b, c, h, w = tensor.shape
    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)

    # Padding: (left, right, top, bottom)
    padding = (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2
    )
    padded = F.pad(tensor, pad=(padding[0], padding[1], padding[2], padding[3]), mode='constant', value=0)
    # print(f"Padding: {padding}, New shape: {padded.shape}, dtype: {padded.dtype}")
    return padded

def get_clip_model(device="cuda:0"):
    import clip
    model, process = clip.load('ViT-B/16', device)

    # model.head = torch.nn.Identity()
    model.eval()
    return Skill("clip", pad_to_clip, model.encode_image, model_forward, None)

if __name__ == "__main__":
    # swin = get_swin()
    # print("swin:\t OK")
    resnet = get_resnet()
    print("resnet:\t OK")
    print(resnet)
    # clip = get_clip_model()
    # print("clip:\t OK")