from transformers import XLMTokenizer, XLMForSequenceClassification
from transformers import XLMTokenizer, XLMModel
import torch.nn as nn


class XLM(nn.Module):
    def __init__(self, number_labels):
        super(XLM, self).__init__()
        self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        self.plm = XLMModel.from_pretrained('xlm-mlm-en-2048')
        self.out = nn.Linear(2048, number_labels)

    def forward(self, ids, mask):
        hidden_states = self.plm(ids, attention_mask=mask, return_dict=False)

        out = self.out(hidden_states[0][:, -1, :])

        return out

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.num_languages
        )