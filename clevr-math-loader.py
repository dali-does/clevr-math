# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLEVR-math dataset of arithmetic operations for visual reasoning"""


import csv
import json
import os
import pandas
import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2208.05358,
  doi = {10.48550/ARXIV.2208.05358},
  url = {https://arxiv.org/abs/2208.05358},
  author = {Lindström, Adam Dahlgren and Abraham, Savitha Sam},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.7; I.2.10; I.2.6; I.4.8; I.1.4},
  title = {CLEVR-Math: A Dataset for Compositional Language, Visual, and Mathematical Reasoning},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
"""

# You can copy an official description
_DESCRIPTION = """\
CLEVR-Math is a dataset for compositional language, visual and mathematical reasoning. CLEVR-Math poses questions about mathematical operations on visual scenes using subtraction and addition, such as "Remove all large red cylinders. How many objects are left?". There are also adversarial (e.g. "Remove all blue cubes. How many cylinders are left?") and multihop questions (e.g. "Remove all blue cubes. Remove all small purple spheres. How many objects are left?").
"""

_HOMEPAGE = "https://people.cs.umu.se/dali/clevr-math"

_LICENSE = "Creative Commons Attribution Share Alike 4.0 International"

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "general": "data/clevr-math.zip",
    "multihop": "data/clevr-math-multihop.zip",
    "images": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
    "test_images": "data/clevr-math-test-images.zip",
}


class ClevrMath(datasets.GeneratorBasedBuilder):
    """Generator for CLEVR-Math used to test compositional language, visual, and mathematical reasoning. Contains two versions of the dataset, one general and one where multihop questions are only available in the test data to provide a zero-shot setting."""

    VERSION = datasets.Version("1.1.0")

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="general", version=VERSION, description="CLEVR-math general"),
        datasets.BuilderConfig(name="multihop", version=VERSION, description="CLEVR-math with single hop in train/val and multihop in test"),
    ]

    DEFAULT_CONFIG_NAME = "general"

    def _info(self):
        features = datasets.Features(
            {
                "template": datasets.Value("string"),
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
                "label": datasets.Value("int64")
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_url = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(data_url)
        image_dir = dl_manager.download_and_extract(_URLS['images'])
        image_dir_test = dl_manager.download_and_extract(_URLS['test_images'])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "clevr-math-train.json"),
                    "imgpath": image_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "clevr-math-test.json"),
                    "imgpath": image_dir_test,
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "clevr-math-val.json"),
                    "imgpath": image_dir,
                    "split": "val",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, imgpath, split):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            json_file = json.load(f)
            df = pandas.json_normalize(json_file['questions'])
            for key, sample in df.iterrows():
                img_path = os.path.join(imgpath,
                                        "CLEVR_v1.0/images",
                                        sample['split'],
                                        sample['image_filename'])

                yield key, {
                    "template": sample['template_filename'][:-5],
                    "id": sample['image_filename'],
                    "question": sample["question"],
                    "image": img_path,
                    "label": sample["answer"]
                }
