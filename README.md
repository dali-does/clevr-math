# CLEVR-Math

This is the repository for CLEVR-Math, and contains the code necessary to generate the dataset and run the experiments presented in the paper:

**[CLEVR-Math: A Dataset for Compositional Language, Visual and Mathematical Reasoning](https://arxiv.org/abs/2208.05358)**
 <br>
 <a href='https://www.umu.se/en/staff/adam-dahlgren-lindstrom/'>Adam Dahlgren Lindström</a>,
 <a href='https://www.oru.se/english/employee/savitha_sam-abraham'>Savitha Sam Abraham</a>,
 <br>

This work will be presented at [IJCLR 2022](ijclr22.doc.ic.ac.uk/).

## Huggingface dataset

The dataset is available through Huggingface at [https://huggingface.co/datasets/dali-does/clevr-math](https://huggingface.co/datasets/dali-does/clevr-math).



# Generating dataset
See the original instructions for generating data with CLEVR: [README_CLEVR.md](README_CLEVR.md)


# Experiments

## CLIP

`python train_clip.py --eopchs 10 --train_samples 10000 --val_samples 1000 --test_samples 2000`

## NS-VQA

```
(ns-vqa) dali@dali2:~/ns-vqa-master/reason$ python tools/run_test.py --run_dir ../data/reason/results --load_checkpoint_path ../data/reason/outputs/5000_samples_reinforce/checkpoint_best.pt --clevr_val_question_path ../data/reason/clevr_h5_resub_final/clevr_test_All_questions.h5 --clevr_val_scene_path ../data/raw/CLEVR_v1.0/scenes/CLEVR_test_scenes.json --save_result_path ../data/reason/results_resub_final.json --clevr_vocab_path ../data/reason/clevr_h5/All/clevr_vocab.json^C
(ns-vqa) dali@dali2:~/ns-vqa-master/reason$ python tools/preprocess_questions.py --input_questions_json ~/cs-home/public_html/clevr-math/mm/clevr-math-train.json --output_h5_file ../data/reason/clevr_h5_resub_final/clevr_train_questions_resub_final.h5 --output_vocab_json ../data/reason/clevr_h5_resub_final/clevr_vocab_comp_resub.json^C
(ns-vqa) dali@dali2:~/ns-vqa-master/reason$ python tools/preprocess_questions.py --input_questions_json ~/cs-home/public_html/clevr-math/mm/clevr-math-val.json  --output_h5_file ../data/reason/clevr_h5_resub_final/clevr_val_intersect_multihop_questions_resub.h5 --input_vocab_json ../data/reason/clevr_h5_resub_final/clevr_vocab_comp_resub.json ^C
(ns-vqa) dali@dali2:~/ns-vqa-master/reason$ python tools/preprocess_questions.py --input_questions_json ~/cs-home/public_html/clevr-math/mm/clevr-math-test.json  --output_h5_file ../data/reason/clevr_h5_resub_final/clevr_test_intersect_multihop_questions_resub_final.h5 --input_vocab_json ../data/reason/clevr_h5_resub_final/clevr_vocab_comp_resub.json ^C
(ns-vqa) dali@dali2:~/ns-vqa-master/reason$ python tools/run_train.py --checkpoint_every 10 --num_iters 100 --run_dir ../data/reason/outputs/resub_final --clevr_train_question_path ../data/reason/clevr_h5_resub_final/clevr_train_11questions_per_family.h5 --clevr_val_question_path ../data/reason/clevr_h5_resub_final/clevr_val_intersect_multihop_questions_resub.h5 --clevr_vocab_path ../data/reason/clevr_h5_resub_final/clevr_vocab_comp_resub.json ^C
(ns-vqa) dali@dali2:~/ns-vqa-master/reason$ python tools/run_train.py --reinforce 1 --learning_rate 1e-5 --checkpoint_every 500 --num_iters 5000 --run_dir ../data/reason/outputs/resub_final --load_checkpoint_path ../data/reason/outputs/resub_final/checkpoint.pt --clevr_val_question_path ../data/reason/clevr_h5_resub_final/clevr_val_intersect_multihop_questions_resub.h5 --clevr_vocab_path ../data/reason/clevr_h5_resub_final/clevr_vocab_comp_resub.json --clevr_train_question_path ../data/reason/clevr_h5_resub_final/clevr_train_questions_resub_final.h5
```
