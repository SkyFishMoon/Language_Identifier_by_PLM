from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from transformers import RobertaForSequenceClassification

class XLMR(nn.Module):
    def __init__(self, number_labels):
        super(XLMR, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.plm = RobertaModel.from_pretrained('roberta-base')
        self.out = nn.Linear(768, number_labels)

    def forward(self, ids, mask):
        hidden_states = self.plm(ids, attention_mask=mask, return_dict=False)

        out = self.out(hidden_states[0][:, -1, :])

        return out

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.num_languages
        )