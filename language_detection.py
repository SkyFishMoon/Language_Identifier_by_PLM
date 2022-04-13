from iso639 import languages
from langdetect import DetectorFactory, detect
import torch
from model.mBert import mBERT
import json
# DetectorFactory.seed = 0
# torch.cuda.set_device(2)
max_length = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('./datasets/UD20/processed/meta.json','r') as load_f:
    load_dict = json.load(load_f)
ckpt = torch.load('./ckpts/mbert/Jan-18_11-30-18/epoch-3.pth')
model = mBERT(load_dict['num_labels']).to(device)
model.load_state_dict(ckpt['state_dict'])

with open("./language_detect_demo.txt", "r") as f:
    text = f.read()
inputs = model.tokenizer.encode_plus(
    text,
    None,
    pad_to_max_length=True,
    truncation="longest_first",
    add_special_tokens=True,
    return_attention_mask=True,
    max_length=max_length,
)
ids = torch.tensor(inputs["input_ids"]).to(device).unsqueeze(0)
# token_type_ids = torch.tensor(inputs["token_type_ids"]).to(device).unsqueeze(0)
mask = torch.tensor(inputs["attention_mask"]).to(device).unsqueeze(0)
logits = model(ids,
               # token_type_ids=token_type_ids,
               mask=mask)
lang_type_code = load_dict['idx_to_lang'][logits.argmax(dim=-1).item()]
lang_type_name = languages.get(alpha2=lang_type_code).name
print("The language used in this document is " + lang_type_name)