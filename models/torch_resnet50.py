import torch
import torchvision.models as models

model = models.resnet50(weights="IMAGENET1K_V2")
model.eval()

example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)
torch.jit.save(traced, "resnet50_traced.pt")
