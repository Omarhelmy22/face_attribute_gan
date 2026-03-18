import torch

data = torch.load("weights/directions/sefa_directions.pt", map_location="cpu", weights_only=False)
torch.save(data["directions"][13], "weights/directions/hair.pt")
print("Saved hair direction (SeFa component 13) to weights/directions/hair.pt")
