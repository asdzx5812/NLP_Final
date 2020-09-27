import os
import unicodedata
import string
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from alphabert_utils import *
from transformers import ElectraTokenizer, ElectraModel

class ElectraForTokenClassification(nn.Module):
    def __init__(self,pretrainfile):
        super(ElectraForTokenClassification, self).__init__()

        self.electra = ElectraModel.from_pretrained(pretrainfile)
        self.config = self.electra.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)
        
        logits_head = logits[:,0]

        output = (logits_head,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            loss = loss_fct(logits_head.view(-1, 2), labels.view(-1))

            output = (loss,) + output

        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)

# 904405

if __name__ == '__main__':
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    model = ElectraForTokenClassification()
    
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=labels)
    
    loss, scores = outputs[:2]    
