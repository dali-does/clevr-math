import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
 
from transformers import AutoModel, AutoConfig

class ClipClassification(nn.Module):
  def __init__(self,checkpoint,num_labels, outdim=512, use_vision=True,
               use_text=True):
    super(ClipClassification,self).__init__()
    self.num_labels = num_labels
    self.outdim = outdim
    self.use_vision = use_vision
    self.use_text = use_text

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
#    self.classifier = nn.Linear(outdim*2,num_labels) # load and initialize weights
    outdim = 0
    if self.use_vision:
        outdim += self.outdim
    if self.use_text:
        outdim += self.outdim
    self.classifier = nn.Sequential(
            nn.Linear(outdim, outdim),
            nn.LayerNorm(outdim),
            nn.GELU(),
            nn.Linear(outdim, num_labels),
        )

  def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

    #Add custom layers
    embs = []
    if self.use_text:
        embs.append(outputs['text_embeds'])
    if self.use_vision:
        embs.append(outputs['image_embeds'])
    emb = torch.concat(embs,dim=1)


    logits = self.classifier(emb) # calculate losses

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    hidden = outputs['text_model_output']['last_hidden_state']

    return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden,attentions=None)
