import torch
from spn.dataset_loader import DirectMaskDataset
from spn.models.soft_pointer_network import SoftPointerNetwork
from spn.tools import show_position_batched

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

limit = 128  # None means unlimited, else 100 would mean to load only the first 100 files
limit = None

DirectMaskDataset.base = ".data"
test_files = DirectMaskDataset.load_csv('test', sa=False)  # do not import calibration sentences
test_dataset = DirectMaskDataset(test_files, limit=limit, device=device)

soft_pointer_model = SoftPointerNetwork(54, 26, 256, device=device)

soft_pointer_model.load(path="spn/trained_weights/position_model-final.pth")

show_position_batched(soft_pointer_model.with_gradient, test_dataset, duration_combined_model=None, plotting=True)
