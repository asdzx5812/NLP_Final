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
from transformers import BertModel, BertTokenizer

class ElectraForTokenClassification(nn.Module):
    def __init__(self,pretrainfile):
        super(ElectraForTokenClassification, self).__init__()

        self.electra = BertModel.from_pretrained(pretrainfile,from_tf=True)
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

class ElectraForQuestionAnswering(nn.Module):
    def __init__(self, pretrainfile,num_labels=8):
        super().__init__()
        self.num_labels = num_labels
        self.electra = BertModel.from_pretrained(pretrainfile)
        self.config = self.electra.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.qa_outputs(discriminator_sequence_output)

#        se_logits = logits.split(1, dim=-1)
#        
#        output = []
#        for se in se_logits:
#            output.append(se.squeeze(-1))

#        if start_positions is not None and end_positions is not None:
#            # If we are on multi-GPU, split add a dimension
#            if len(start_positions.size()) > 1:
#                start_positions = start_positions.squeeze(-1)
#            if len(end_positions.size()) > 1:
#                end_positions = end_positions.squeeze(-1)
#            # sometimes the start/end positions are outside our model inputs, we ignore these terms
#            ignored_index = start_logits.size(1)
#            start_positions.clamp_(0, ignored_index)
#            end_positions.clamp_(0, ignored_index)
#
#            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#            start_loss = loss_fct(start_logits, start_positions)
#            end_loss = loss_fct(end_logits, end_positions)
#            total_loss = (start_loss + end_loss) / 2
#            output = (total_loss,) + output
#
#        output += discriminator_hidden_states[1:]

        return logits  # (loss), scores, (hidden_states), (attentions)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ElectraForQuestionAnswering(pretrainfile="bert-base-uncased")
    
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    
    encoding = tokenizer.encode_plus(question, text)
    input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
#    logits = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    outputs = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    start_scores, end_scores = outputs[0],outputs[1]
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

    print('answer : ', answer)
