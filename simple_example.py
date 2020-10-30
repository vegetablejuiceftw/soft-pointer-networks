from models.soft_pointer_network import SoftPointerNetwork

soft_pointer_model = SoftPointerNetwork(54, 26, 256, device="cpu")

soft_pointer_model.load(path="trained_weights/position_model-final.pth")

print(soft_pointer_model)
print(soft_pointer_model.with_weights.with_argmax)
