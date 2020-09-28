from models.soft_pointer_network import SoftPointerNetwork
from .training_helpers import train, MaskedMSE, position_gradient_trainer

soft_pointer_model = SoftPointerNetwork(54, 26, 256, device='cpu')

soft_pointer_model.load(path='trained_weights/position_model-final.pth')

print(soft_pointer_model)
print(soft_pointer_model.with_weights.with_argmax)

train(
    soft_pointer_model.with_gradient,
    10,
    None,
    None,
    loss_function=MaskedMSE(),
    train_function=position_gradient_trainer,
    lr_decay=0.951, lr= 0.000161 * 0.131 * .10, weight_decay=1e-05 * 14)
