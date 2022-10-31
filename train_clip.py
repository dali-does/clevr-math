import torch
import pandas
import json
import argparse
import logging
import os

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoConfig
from transformers import Trainer, TrainingArguments

from transformers.modeling_outputs import SequenceClassifierOutput

from datasets import load_dataset, load_metric, DownloadConfig, load_from_disk, DatasetDict

import datasets

from sklearn import metrics

datasets.config.MAX_TABLE_NBYTES_FOR_PICKLING =  500 << 20
datasets.config.IN_MEMORY_MAX_SIZE = 30000000000

class ClipClassification(nn.Module):
  def __init__(self,checkpoint,num_labels, outdim=512):
    super(ClipClassification,self).__init__()
    self.num_labels = num_labels
    self.outdim = outdim

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(outdim*2,num_labels) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

    #Add custom layers
    text_emb = outputs['text_embeds']
    image_emb = outputs['image_embeds']
    emb = torch.concat([text_emb,image_emb],dim=1)
    emb = self.dropout(emb)

    logits = self.classifier(emb) # calculate losses

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    hidden = outputs['text_model_output']['last_hidden_state']

    return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden,attentions=None)


if __name__ == "__main__":
  #TODO how does learning on only question-answer pairs vs. also seeing the logic program compare?
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int)
  parser.add_argument('--n_samples', type=int)
  parser.add_argument('--samples_path', type=str, default='samples/')
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--model_path', type=str)
  parser.add_argument('--save_to_disk', type=str)
  parser.add_argument('--train_samples', type=int, default=10000)
  parser.add_argument('--val_samples', type=int, default=2000)
  parser.add_argument('--test_samples', type=int, default=5000)
  parser.add_argument('--name', type=str, default='general')
  parser.add_argument('--cachedir', type=str, default=None)
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
      logging.StreamHandler()
    ]
  )

  logging.info("Loading CLIP")
  model_path = "openai/clip-vit-base-patch32"
  model = CLIPModel.from_pretrained(model_path)
  model_path = "openai/clip-vit-base-patch32"


  logging.info("Loading dataset")
  if args.data_path:
    dataset = load_from_disk(args.data_path)
  else:
    dl_config = DownloadConfig(resume_download=True, num_proc=8,
            force_download=True)
    logging.info('Loading training data')
    dataset_train = load_dataset('dali-does/clevr-math',
            name=args.name,
            download_config=dl_config,
            split='train[:{}]'.format(args.train_samples),
            cache_dir='/scratch/dali/clevrcache/')
    logging.info('Loading validation data')
    dataset_val = load_dataset('dali-does/clevr-math',
            name=args.name,
            download_config=dl_config,
            split='validation[:{}]'.format(args.val_samples),
            cache_dir='/scratch/dali/clevrcache/')
    logging.info('Loading test data')
    dataset_test = load_dataset('dali-does/clevr-math',
            name=args.name,
            download_config=dl_config,
            split='test[:{}]'.format(args.test_samples),
            cache_dir='/scratch/dali/clevrcache/')
    logging.info('Dataset loaded')

    dataset = DatasetDict({
      'train':dataset_train,
      'validation':dataset_val,
      'test':dataset_test
    })

    logging.info('Selecting subsets')
    dataset['train'] = dataset['train'].select(range(args.train_samples))
    dataset['validation'].select(range(args.val_samples))
    dataset['test'].select(range(args.test_samples))

    logging.info('Loading CLIP')
    #TODO convert CLEVR images offline
    extractor = CLIPProcessor.from_pretrained(model_path)
    def transform_tokenize(e):
      e['image'] = [image.convert('RGB') for image in e['image']]
      return extractor(text=e['question'],
                               images=e['image'],
                               padding=True)

    #cache_file_names = {
    #        'train': TODO ,
    #        'validation': TODO,
    #        'test': TODO,
    #        }
    logging.info('Transforming dataset')
    dataset = dataset.map(transform_tokenize,
                          #        cache_file_names=cache_file_names,
                          #        keep_in_memory=True,
                          batched=True,
                          num_proc=8,
                          padding='max_length'
                          )

    if args.save_to_disk:
      dataset.save_to_disk(args.save_to_disk)

  logging.info('Filtering datasets')
  dataset_multihop = dataset.filter(lambda e:
          e['template'].startswith('subtraction-multihop'), num_proc=4)
  dataset_adversarial = dataset.filter(lambda e:
          e['template'].startswith('adversarial'), num_proc=4)
  dataset_subtraction = dataset.filter(lambda e:
          e['template'].startswith('subtraction'), num_proc=4)
  dataset_addition = dataset.filter(lambda e:
          e['template'].startswith('addition'), num_proc=4)


  metric = load_metric('accuracy')
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[:-1], axis=-1)[0]
    return metric.compute(predictions=predictions, references=labels)

  logging.info('Loading model')
  model = ClipClassification(model_path, 11)

  logging.info("Creating trainer")
  training_args = TrainingArguments("test_trainer",
                                    num_train_epochs=args.epochs,
                                    per_device_train_batch_size=32,
                                    fp16=True,
                                    dataloader_num_workers=8,
                                    dataloader_pin_memory=8,
                                    gradient_accumulation_steps=1,
                                    save_strategy='no',
                                    evaluation_strategy='epoch',
                                    eval_steps=1)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
  )

  logging.info("Training model")

  training_metrics = trainer.train()
  logging.info(training_metrics)
  def get_sample_ids(y_true, y_pred,n=10, matching=True,rand=True):
      num_items = len(y_true)

      ids = []
      counter = 0
      while len(ids) < n and counter < 1000:
          counter += 1
          i = np.random.randint(0, num_items)
          is_matching = y_true[i] == y_pred[i]
          if matching == is_matching:
              ids.append(i)
      
      return ids

  for test_data in [dataset, dataset_subtraction, dataset_addition, dataset_adversarial, dataset_multihop]:
    predictions, labels, test_metrics = trainer.predict(test_data['test'])

    y_true = labels #test_data['test']['label']
    y_pred = np.argmax(predictions[:-1], axis=-1)[0]

    # sample n_samples correct and 10 wrong answers for closer inspection
    n_samples = args.n_samples

    correct_samples = get_sample_ids(y_true, y_pred, n=n_samples)
    incorrect_samples = get_sample_ids(y_true, y_pred, n=n_samples, matching=False)

    logging.info('Incorrect answers')
    for i in range(n_samples):
        sample_index = incorrect_samples[i]
        image_index = sample_index
        sample = test_data['test'][sample_index]
        if 'id' in sample:
            image_index = sample['id']
        print('Question {}(image {}) ={}= was incorrectly answered with {} instead of {}'
                .format(sample_index, image_index, sample['question'],
                    y_pred[sample_index], sample['label']))
        sample['image'].save('{}/incorrect/{}.png'.format(args.sample_path, image_index))

    logging.info('Correct answers')
    for i in range(n_samples):
        sample_index = correct_samples[i]
        image_index = sample_index
        sample = test_data['test'][sample_index]
        if 'id' in sample:
            image_index = sample['id']
            print("has id")
        print(sample['question'])
        print('Question {}(image {}) ={}= was correctly answered with {}'
                .format(sample_index, image_index, sample['question'], sample['label']))
        sample['image'].save('{}/correct/{}.png'.format(args.sample_path, image_index))


    confusion_matrix = metrics.confusion_matrix(y_true, y_pred,
            labels=[0,1,2,3,4,5,6,7,8,9,10])
    print(confusion_matrix)
    logging.info(test_metrics)
