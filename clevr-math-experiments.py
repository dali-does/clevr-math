import torch
import datasets
import argparse
import logging
import transformers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from transformers import DataCollatorWithPadding
from transformers import CLIPModel, CLIPProcessor
from transformers import Trainer, TrainingArguments
from transformers import ViltForQuestionAnswering, ViltProcessor
from datasets import load_metric, concatenate_datasets
from datasets import load_dataset, DownloadConfig, load_from_disk, DatasetDict
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

from clip import ClipClassification

datasets.config.MAX_TABLE_NBYTES_FOR_PICKLING =  500 << 20
datasets.config.IN_MEMORY_MAX_SIZE = 30000000000

def parse_clevr_math_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=1)
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
  parser.add_argument('--save_to_tensorboard', type=bool, default=True)
  return parser.parse_args()

def compute_attribute_stats(dataset, indices):
    stats = {}
    # 1-gram
    # 2-gram
    return stats

def load_vilt(model_path, num_labels):
    id2label = {}
    label2id = {}
    for i in range(num_labels):
        id2label[i] = i
        label2id[i] = i
    return ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 num_labels=num_labels,
                                                 id2label=id2label,
                                                 label2id=label2id)

def mix_datasets(a, b, ratio=0.1, shuffle=False, split='train'):
  assert(ratio<=1.0)
  assert(ratio>0)
  assert(len(a) == len(b))
  if shuffle:
    a = a.shuffle()
    b = b.shuffle()
  a = a.select(range(len(a)*ratio))
  b = b.select(range(len(b)*(1-ratio)))
  return concatenate_datasets([a, b])

def preprocess_data(dataset, model_path):
  extractor = CLIPProcessor.from_pretrained(model_path)
  def transform_tokenize(e):
    e['image'] = [image.convert('RGB') for image in e['image']]
    return extractor(text=e['question'],
                     images=e['image'],
                     padding='max_length')
  dataset = dataset.map(transform_tokenize,
                        #        cache_file_names=cache_file_names,
                        #        keep_in_memory=True,
                        batched=True,
                        num_proc=8
                        )
  return dataset

def onehot_label(label):
    enc = np.zeros(11)
    enc[label] = 1
    return enc

def preprocess_data_otf(dataset, extractor):
  def transform_tokenize(e):
    e['image'] = [image.convert('RGB') for image in e['image']]
    extracts = extractor(text=e['question'],
                     images=e['image'],
                     padding='max_length',
                     return_tensors='pt')
    extracts['label'] = e['label']#[onehot_label(label) for label in e['label']]
    return extracts
  return dataset.with_transform(transform_tokenize)

def create_metric(metric):
  metric = load_metric(metric)
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[:-1], axis=-1)[0]
    return metric.compute(predictions=predictions, references=labels)
  return compute_metrics

def setup_trainer(model, train_dataset, eval_dataset, metric, args):
  callbacks = []
  if args.save_to_tensorboard:
    callbacks.append('tensorboard')
      
  training_args = TrainingArguments("clip_trainer",
                                    num_train_epochs=args.epochs,
                                    per_device_train_batch_size=64,
                                    fp16=True,
                                    dataloader_num_workers=8,
                                    dataloader_pin_memory=8,
                                    gradient_accumulation_steps=1,
                                    save_strategy='no',
                                    report_to=callbacks,
                                    evaluation_strategy='epoch',
                                    remove_unused_columns=False,
                                    eval_steps=1)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metric,
  )
  return trainer

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

def get_samples(dataset, y_true, y_pred, n_samples, matching=False):
  df_samples = pd.DataFrame(columns=['correct',
                                         'sample_index', 
                                         'image_index',
                                         'question',
                                         'prediction',
                                         'label',
                                         'image'])

  samples = get_sample_ids(y_true, y_pred, n=n_samples, matching=matching)

  for i in range(n_samples):
    sample_index = samples[i]
    sample = dataset[sample_index]
    if 'id' in sample:
      df_samples.loc[len(df_samples)] = [False, 
                                         sample_index,
                                         sample['id'],
                                         sample['question'], 
                                         y_pred[sample_index],
                                         sample['label'],
                                         sample['image']]
  return df_samples



def evaluate_model(trainer, test_data, identifier, n_samples=10, sample=False):
  predictions, labels, test_metrics = trainer.predict(test_data)

  y_true = labels 
  y_pred = np.argmax(predictions[:-1], axis=-1)[0]

  if sample:
      df_incorrect_samples = get_samples(test_data, y_true, y_pred, n_samples)
      df_correct_samples = get_samples(test_data, y_true, y_pred, n_samples, matching=True)
    
      df_samples = pd.concat(df_incorrect_samples, df_correct_samples)
      df_samples.to_pickle('samples-{}-{}.pkl'.format(pd.Timestamp.now(), identifier))


  confusion_matrix = metrics.confusion_matrix(y_true, y_pred,
                                              labels=[0,1,2,3,4,5,6,7,8,9,10])
  print(confusion_matrix)
  return test_metrics


def experiment():
  dataset = load_dataset('dali-does/clevr-math')
  dataset = preprocess_data(dataset)

  metric = create_metric('accuracy')
  trainer = setup_trainer(model_path, dataset['train'], dataset['validation'],
                          metric, args)

  training_metrics = trainer.train()

  evaluate_model(trainer, dataset['test'], 'general')

def cogent_experiment():
  dataset = load_dataset('dali-does/clevr-math', 'cogent')
  #TODO Use cached
  dataset = preprocess_data(dataset)

  metric = create_metric('accuracy')

  ratios = [0.0, 0.1, 0.2, 0.5]

  for ratio in ratios:
    dataset_mixed = mix_datasets(dataset['trainB'], dataset['trainA'], ratio, split='train')
    trainer = setup_trainer(model_path, dataset_mixed['train'],
                            dataset['valA'], metric, args)
    training_metrics = trainer.train()
    evaluate_model(trainer, dataset['testB'], '{}-cogent'.format(ratio))


if __name__ == "__main__":
  args = parse_clevr_math_arguments()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
      logging.StreamHandler()
    ]
  )

  model_path = "openai/clip-vit-base-patch32"
  #model_path = "dandelin/vilt-b32-finetuned-vqa"
  processor = CLIPProcessor.from_pretrained(model_path)
  #processor = ViltProcessor.from_pretrained(model_path)

  logging.info("Loading dataset")
  if args.data_path:
    dataset = load_from_disk(args.data_path)
  else:
    dl_config = DownloadConfig(resume_download=True, num_proc=8,
            force_download=True)
    logging.info('Loading training data')
    dataset_train = load_dataset('dali-does/clevr-math',
            name=args.name,
            revision='comp-gen',
            download_config=dl_config,
            split='trainA[:{}]'.format(args.train_samples))#,
            #cache_dir=args.cachedir)
    logging.info('Loading validation data')
    dataset_val = load_dataset('dali-does/clevr-math',
            name=args.name,
            revision='comp-gen',
            download_config=dl_config,
            split='valA[:{}]'.format(args.val_samples))#,
            #cache_dir=args.cachedir)
    logging.info('Loading test data')
    dataset_testA = load_dataset('dali-does/clevr-math',
            name=args.name,
            revision='comp-gen',
            download_config=dl_config,
            split='testA[:{}]'.format(args.test_samples))#,
            #cache_dir=args.cachedir)
    logging.info('Dataset loaded')
    dataset_testB = load_dataset('dali-does/clevr-math',
            name=args.name,
            revision='comp-gen',
            download_config=dl_config,
            split='testB[:{}]'.format(args.test_samples))#,
            #cache_dir=args.cachedir)

    dataset_train = dataset_train.filter(lambda e:
                                         'sphere' not in e['question'],
                                        num_proc=8)
    dataset_val = dataset_val.filter(lambda e:
                                         'sphere' not in e['question'],
                                        num_proc=8)
    dataset_testB = dataset_testB.filter(lambda e:
                                         'sphere' not in e['question'],
                                        num_proc=8)
    dataset = DatasetDict({
      'train':dataset_train,
      'validation':dataset_val,
      'test':dataset_testB,
      'testA':dataset_testA,
      'testB':dataset_testB
    })


    dataset = preprocess_data_otf(dataset, processor)
    if args.save_to_disk:
      dataset.save_to_disk(args.save_to_disk)

  metric = create_metric('accuracy')
  #model = ViltForQuestionAnswering.from_pretrained(model_path)
  #model = load_vilt(model_path, 11)
  model = ClipClassification(model_path, 11, use_text=False)
  trainer = setup_trainer(model, dataset['train'],
                                   dataset['validation'], metric, args)

  training_metrics = trainer.train()

  #trainer.evaluate(dataset['test'])
  test_metrics = evaluate_model(trainer, dataset['testA'], 'general')
  logging.info(test_metrics)
  test_metrics = evaluate_model(trainer, dataset['testB'], 'general')
  logging.info(test_metrics)
