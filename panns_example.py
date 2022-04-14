import torch

from load import load_csv, load_files
from spn.models.panns import Cnn14_DecisionLevelAtt

model = Cnn14_DecisionLevelAtt(
    sample_rate=16000, window_size=1024,
    hop_size=256, mel_bins=64, fmin=50, fmax=14000,
    classes_num=527)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.att_block.activation = "linear"

model_data = torch.load('Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))['model']
# print(model_data.keys())
# for key in tuple(model_data):
#     if 'att_block' in key:
#         del model_data[key]
model.load_state_dict(model_data, strict=False)

limit = 8
base = ".data"
test_files = load_csv(f"{base}/test_data.csv", sa=False)[:limit]
test_files = load_files(base, test_files)

print(test_files[0].audio.shape)
print(test_files[0].features_spectogram.shape)

audio = torch.Tensor(test_files[0].audio).unsqueeze(0)
res = model(audio, None)

print(res['clipwise_output'].shape)
print(res['framewise_output'].shape)
