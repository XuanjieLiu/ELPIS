import torch
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
from shared import DEVICE

DINO_URL = (
    "https://dl.fbaipublicfiles.com/dino/"
    "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
)
# Pin the official checkpoint to the weights used by the retained experiments.
DINO_CACHE_NAME = "dino_deitsmall8_pretrain-55c8b267.pth"

# Define a function to load the DINO model
def load_dino_vit_s8(checkpoint_path=None):
    # Load the ViT-S/8 model structure
    model = VisionTransformer(
        img_size=224,  # DINO typically trains on 224x224 images
        patch_size=8,  # S/8 means patch size is 8
        embed_dim=384,  # ViT-S has an embedding dimension of 384
        depth=12,  # Number of transformer blocks
        num_heads=6,  # Number of attention heads
        num_classes=0,  # No classification head
    )

    # Load the checkpoint
    if checkpoint_path is None:
        checkpoint = torch.hub.load_state_dict_from_url(
            DINO_URL, map_location=DEVICE, file_name=DINO_CACHE_NAME,
            check_hash=True,
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("teacher", checkpoint)  # DINO checkpoints usually store the teacher's state dict

    # Remove `module.` prefix from keys if it exists
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    model.eval()

    return model

# Define a transform pipeline for your images
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
])


# Assume you have a dataloader named `dataloader`, convert images to their features
def evaluate_images(model, dataloader):
    features = []
    with torch.no_grad():  # Disable gradient computation for inference
        for images, _ in dataloader:
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = model(images)  # Extract features
            features.append(outputs.cpu())  # Store features on CPU

    # Concatenate all features into a single tensor
    features = torch.cat(features, dim=0)
    return features


if __name__ == "__main__":
    # Load the DINO model
    # Load your DINO model
    model = load_dino_vit_s8()
    print("DINO model loaded successfully!")

    # Load your dataset
    # dataloader = ...

    # Extract features from the dataset
    # features = evaluate_images(dino_model, dataloader)
