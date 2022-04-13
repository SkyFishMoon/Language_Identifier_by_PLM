from torch.utils.data import Dataset
import pandas as pd
import torch


class LDDataset(Dataset):
    def __init__(self, tokenizer, path, max_length):
        super(LDDataset, self).__init__()
        # self.root_dir = root_dir
        self.all_data = pd.read_csv(path).dropna(axis=0, how='any')
        self.tokenizer = tokenizer
        self.target = self.all_data.iloc[:, 1]
        self.max_length = max_length

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        text1 = self.all_data.iloc[index, 1]

        inputs = self.tokenizer.encode_plus(
            text1,
            None,
            pad_to_max_length=True,
            truncation="longest_first",
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.all_data.iloc[index, 2], dtype=torch.long)
        }

