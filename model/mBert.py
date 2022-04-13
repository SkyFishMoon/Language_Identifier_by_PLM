import torch.nn as nn
import transformers
from transformers import BertTokenizer


class mBERT(nn.Module):
    def __init__(self, number_labels):
        super(mBERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.plm = transformers.BertModel.from_pretrained("bert-base-multilingual-cased")
        self.out = nn.Linear(768, number_labels)

    def forward(self, ids, mask):
        _, o2 = self.plm(ids, attention_mask=mask, return_dict=False)

        out = self.out(o2)

        return out

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.num_languages
        )

# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-multilingual-cased", # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels=2, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions=False, # Whether the model returns attentions weights.
#     output_hidden_states=False, # Whether the model returns all hidden-states.
# )
