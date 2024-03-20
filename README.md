# A Hierarchical Encoding-Decoding Scheme for Abstractive Multi-document Summarization
**Authors**: Chenhui Shen, Liying Cheng, Xuan-Phi Nguyen, Yang You and Lidong Bing

This repository contains code and related resources of our paper ["A Hierarchical Encoding-Decoding Scheme for Abstractive Multi-document Summarization"](https://aclanthology.org/2023.findings-emnlp.391.pdf).

<!-- :star2: Check out this awesome [[demo]](https://huggingface.co/spaces/joaogante/contrastive_search_generation) generously supported by Huggingface ([@huggingface](https://github.com/huggingface) :hugs:) which compares contrastive search with other popular decoding methods. Many thanks to Huggingface :hugs:!  -->


****
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@inproceedings{shen2023hierencdec,
  title={A Hierarchical Encoding-Decoding Scheme for Abstractive Multi-document Summarization},
  author={Shen, Chenhui and Cheng, Liying and Nguyen, Xuan-Phi and Bing, Lidong and You, Yang},
  booktitle={Findings of EMNLP},
  url={"https://arxiv.org/abs/2305.08503"},
  year={2023}
}

```

<!-- ****

### News:
* [2022/10/26] Some content

**** -->

<span id='all_catelogue'/>

### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#reproduce_examples'>2. Running our code</a>
    * <a href='#pre-requisites'>2.1. Pre-requisites</a>
    * <a href='#summarization'>2.2. Commands to reproduce our results</a>
        * <a href='#bart'>2.2.1. Reproduce HierEncDec on BART </a>
        * <a href='#baselines'>2.2.2. Reproduce our baselines</a>
    * <a href='#analysis'>2.3. Attention Analysis </a>

    
****

<span id='introduction'/>

# 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

Pre-trained language models (PLMs) have accomplished impressive achievements in abstractive single-document summarization (SDS). However, such benefits may not be readily extended to muti-document summarization (MDS), where the interactions among documents are more complex. Previous works either design new architectures or new pre-training objectives for MDS, or apply PLMs to MDS without considering the complex document interactions. While the former does not make full use of previous pre-training efforts and may not generalize well across multiple domains, the latter cannot fully attend to the intricate relationships unique to MDS tasks. In this paper, we enforce hierarchy on both the encoder and decoder and seek to make better use of a PLM to facilitate multi-document interactions for the MDS task. We test our design on 10 MDS datasets across a wide range of domains. Extensive experiments show that our proposed method can achieve consistent improvements on all these datasets, outperforming the previous best models, and even achieving better or competitive results as compared to some models with additional MDS pre-training or larger model parameters.

****


<span id='reproduce_examples'/>


# 2. Running our Code
The data can be downloaded at <a href="https://drive.google.com/file/d/1F8W96ZE244YJPZQjKNTpA72jUd5pInwM/view?usp=drive_link">HierEncDec_data.zip</a>.
Unzip this folder to data/ and store it under the root directory.
Alternatively, you may use your own dataset formated as ``Doc 1 <REVBREAK> Doc 2 <REVBREAK> ... <REVBREAK> Doc n``, where `` <REVBREAK> `` are the separator between documents.
The exact locations where we download existing datasets are provided in Appendix B of our paper.
Note that to reporduce the <a href="#baselines">PRIMERA</a> results, you need to use ``<doc-sep>'' for the separator token instead.

For the downloaded ``data/``,  our new datasets are organized as follows:
- the MReD+ data are under the ``mred/`` folder, ending with ``_rebuttal.json``.
- the 4 Wikipedia domains data are stored under the folders of ``Film/``, ``MeanOfTransportation/``, ``Software/``, and ``Town/`` respectively.

<span id='pre-requisites'/>

## 2.1. Pre-requisites: <a href='#all_catelogue'>[Back to Top]</a>
We use conda evironments.
```yaml
conda create --prefix <path_to_env> python=3.7
conda activate <path_to_env>
# install torch according to your cuda version, for instance:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

<span id='summarization'/>

## 2.2. Commands to reproduce our results: <a href='#all_catelogue'>[Back to Top]</a>

<!-- For all experiments below, please download our processed data from <a href="">here</a>. -->
<!-- Unzip the downloaded data and place all data folders under the root folder named ```/data```. -->

<span id='bart'/>

To quickly test the code for the following sections, add the following flags
```yaml
--max_steps 10 --max_train_samples 10 --max_eval_samples 10 --max_predict_samples 10
```

__NOTE__: 
1. To replicate our results, you need to run on A100 (80G). Alternatively, for running on __V100__, truncate source input further by setting a smaller value of ``max_source_length`` (e.g. __1024__, __2048__) to __avoid OOM error__, but _this differs from the setting of __4096__ in our paper_.
2. Currently our code only __supports for batch size of 1__ (a larger number will lead to OOM errors anyway), so it is __important__ to set 
``--per_device_train_batch_size=1 --per_device_eval_batch_size=1``.

### 2.2.1. Reproduce results on BART: <a href='#all_catelogue'>[Back to Top]</a>

To reproduce our BART+HED on MReD, run the following command:
```yaml
CUDA_VISIBLE_DEVICES=0 python run_summarization.py --output_dir results/bart_hed_mred --model_name_or_path facebook/bart-large --do_train --do_predict --train_file data/mred/train.csv --test_file data/mred/test.csv --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --seed 0 --max_source_length 4096 --max_target_length 1024 --save_steps 500 --save_strategy steps --save_total_limit 3 --num_train_epochs 3 --max_steps 10500 --enc_cross_doc --doc_dec
```
Specifically, the flags ``--enc_cross_doc`` enables the hierarchical encoder, whereas ``--doc_dec`` enables hieararchical decoder. 

For other datasets, set ``--max_steps`` to the following values, and use ``--per_passage_source_length_limit`` for the first 3 datasets (see more explainations in Section 5.3 of <a href="https://arxiv.org/abs/2305.08503">our paper</a>).
* Mutlinews: 130000, use additional flag ``--per_passage_source_length_limit``
* WCEP: 15500, use additional flag ``--per_passage_source_length_limit``
* Multi-Xscience: 90000, use additional flag ``--per_passage_source_length_limit``
* Rotten Tomatoes: 4500
* MReD: 10500
* MReD+: 10500
* WikiDomains-Film: 85000
* WikiDomains-MeanOfTransportation: 20000
* WikiDomains-Town: 37000
* WikiDomains-Software: 35000

<!-- For Multinews and WCEP, we follow <a href="https://github.com/allenai/PRIMER"> PRIMERA </a> to truncate source by limiting each document to an equal size of length (i.e. truncate the end of each document to satisfy the source length limit). Thus, we use an additional flag ``--per_passage_source_length_limit``. For other datasets, source truncation is simply done by truncating the end of the combined source documents. -->

<!-- This is because the front of the passage is very important for news articles whereas the end of the passage matters less.  -->

For ablation (see Tab.4) settings, 

```yaml
# using <s> components only
CUDA_VISIBLE_DEVICES=0 python run_summarization.py --output_dir results/bart_hed_mred --model_name_or_path facebook/bart-large --do_train --do_predict --train_file data/mred/train.csv --test_file data/mred/test.csv --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --seed 0 --max_source_length 4096 --max_target_length 1024 --save_steps 500 --save_strategy steps --save_total_limit 3 --num_train_epochs 3 --max_steps 10500

# NOTE: For all following settings, simply add the following flags on top of the above command

# run on the pre-trained BART without any structural modifications
--use_original_bart

# using <s> and HAE
--enc_cross_doc --no_posres_only

# using <s>, HAE and PR
--enc_cross_doc

# using <s>, HAE, and HAD
--enc_cross_doc --no_posres_only --doc_dec

# using <s>, HAE, HAD, and PR (This is basically our full HED setting)
--enc_cross_doc --doc_dec

```


<span id='baselines'/>

### 2.2.1. Reproduce our baselines (Table 1 upper section): <a href='#all_catelogue'>[Back to Top]</a>


* to reproduce the LED results, run:
    ```yaml
    python finetune_led.py --output_dir results/led_mred --model_name_or_path allenai/led-large-16384 --do_train --do_predict --train_file data/mred/train.csv --test_file data/mred/test.csv --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --seed 0 --max_source_length 4096 --max_target_length 1024 --save_steps 500 --save_strategy steps --save_total_limit 3 --num_train_epochs 3 --max_steps 10500
    ```

* to reproduce the LongT5 results, run:
    
    ```yaml
    python finetune_longt5.py --output_dir results/longt5_base_mred --source_prefix 'summarize: ' --model_name_or_path google/long-t5-tglobal-base --do_train --do_predict --train_file data/mred/train.csv --test_file data/mred/test.csv --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --seed 0 --max_source_length 4096 --max_target_length 1024 --save_steps 500 --save_strategy steps --save_total_limit 3 --num_train_epochs 3 --max_steps 10500
    ````



* to reproduce the PRIMERA results, kindly follow <a href="https://github.com/allenai/PRIMER"> this GitHub repo </a>.

* to reproduce the BigBird results, run:

    ```yaml
    CUDA_VISIBLE_DEVICES=0 python finetune_bigbird.py --output_dir results/bigbird_mred --model_name_or_path google/bigbird-pegasus-large-arxiv --do_train --do_predict --train_file data/mred/train.csv --test_file data/mred/test.csv --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --seed 0 --max_source_length 4096 --max_target_length 1024 --save_steps 500 --save_strategy steps --save_total_limit 3 --num_train_epochs 3 --max_steps 10500
    ```

<span id='analysis'/>

# 2.3. Attention Analysis: <a href='#all_catelogue'>[Back to Top]</a>

To conduct attention analysis, run 
```yaml
CUDA_VISIBLE_DEVICES=0 python run_summarization.py --model_name_or_path results/<your_trained_model_name> --output_dir results/<your_preferred_save_dir> --do_predict --test_file data/mred/test.csv --overwrite_output_dir --per_device_eval_batch_size=1 --predict_with_generate --max_source_length 4096 --max_target_length 1024 --max_predict_samples 200 --enc_cross_doc --doc_dec --model_analysis --analyze_self_attn --analyze_cross_attn --model_analysis_file mred_hed_attn_analysis.txt 
```
Specifically, ``model_analysis`` must be enabled, whereas ``analyze_self_attn`` and (or) ``analyze_cross_attn`` can be used together to conduct the corresponding encoder self-attention and (or) decoder cross-attention analysis.



