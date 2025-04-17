---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:3552
- loss:TripletLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: "9.\tWhat is an online learning system?\n10.\tWhat is out-of-core\
    \ learning?\n11.\tWhat type of learning algorithm relies on a similarity measure\
    \ to make predictions?\n12.\tWhat is the difference between a model parameter\
    \ and a learning algorithm’s hyperparameter?\n13.\tWhat do model-based learning\
    \ algorithms search for? What is the most common strategy they use to succeed?\
    \ How do they make predictions?\n14.\tCan you name four of the main challenges\
    \ in Machine Learning?\n15.\tIf your model performs great on the training data\
    \ but generalizes poorly to new instances, what is happening? Can you name three\
    \ possible solutions?\n16.\tWhat is a test set, and why would you want to use\
    \ it?\n17.\tWhat is the purpose of a validation set?\n18.\tWhat is the train-dev\
    \ set, when do you need it, and how do you use it?\n19.\tWhat can go wrong if\
    \ you tune hyperparameters using the test set? Solutions to these exercises are\
    \ available in Appendix A."
  sentences:
  - supervised learning
  - hyperparameters
  - stacked generalization
- source_sentence: 'possible values of x, you always get 1; but if you integrate the
    likelihood function over all possible values of θ, the result can be any positive
    value.

    Given a dataset X, a common task is to try to estimate the most likely values
    for the model parameters. To do this, you must find the values that maximize the
    likelihood function, given X. In this example, if you have observed a single instance
    x=2.5, the maximum likelihood estimate (MLE) of θ is θ=1.5. If a prior probability
    distribution g over θ exists, it is possible to take it into account by maximizing
    ℒ(θ|x)g(θ) rather

    than just maximizing ℒ(θ|x). This is called maximum a-posteriori (MAP) estimation.
    Since MAP constrains the parameter values, you can think of it as a regularized
    version of MLE.

    Notice that maximizing the likelihood function is equivalent to maximizing its
    logarithm (represented in the lower-righthand plot in Figure 9-20). Indeed the
    logarithm is a strictly increasing function, so if θ maximizes the log likelihood,
    it also maximizes the likelihood. It turns out that it is generally easier to
    maximize the log likelihood. For example, if you observed several independent
    instances x(1) to x(m), you would need to find the value of θ that maximizes the
    product of the individual likelihood functions. But it is equivalent, and much
    simpler, to maximize the sum (not the product) of the log likelihood functions,
    thanks to the magic of the logarithm which converts products into sums: log(ab)=log(a)+log(b).

    Once you have estimated θ, the value of θ that maximizes the likelihood function,
    then you are ready to compute L = ℒ θ, X , which is the value used to compute
    the AIC and BIC; you can think of it as a measure of how well the model fits the
    data.'
  sentences:
  - p
  - project goals
  - Extra-Trees classifier
- source_sentence: "3.\tTackle the Titanic dataset. A great place to start is on Kaggle.\n\
    4.\tBuild a spam classifier (a more challenging exercise):\n•\tDownload examples\
    \ of spam and ham from Apache SpamAssassin’s public datasets.\n•\tUnzip the datasets\
    \ and familiarize yourself with the data format.\n•\tSplit the datasets into a\
    \ training set and a test set.\n•\tWrite a data preparation pipeline to convert\
    \ each email into a feature vector. Your preparation pipeline should transform\
    \ an email into a (sparse) vector that indicates the presence or absence of each\
    \ possible word. For example, if all emails only ever contain four words, “Hello,”\
    \ “how,” “are,” “you,” then the email “Hello you Hello Hello you” would be converted\
    \ into a vector [1, 0, 0, 1] (meaning [“Hello” is present, “how” is absent, “are”\
    \ is absent, “you” is present]), or [3, 0, 0, 2] if you prefer to count the number\
    \ of occurrences of each word.\nYou may want to add hyperparameters to your preparation\
    \ pipeline to control whether or not to strip off email headers, convert each\
    \ email to lowercase, remove punctuation, replace all URLs with “URL,” replace\
    \ all numbers with “NUMBER,” or even perform stemming (i.e., trim off word endings;\
    \ there are Python libraries available to do this).\nFinally, try out several\
    \ classifiers and see if you can build a great spam classifier, with both high\
    \ recall and high precision.\nSolutions to these exercises can be found in the\
    \ Jupyter notebooks available at https:// github.com/ageron/handson-ml2."
  sentences:
  - Dueling DQN
  - Decision Stumps
  - precision
- source_sentence: 'Figure 3-6. This ROC curve plots the false positive rate against
    the true positive rate for all possible thresholds; the red circle highlights
    the chosen ratio (at 43.68% recall)

    One way to compare classifiers is to measure the area under the curve (AUC). A
    perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier
    will have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute
    the ROC AUC:

    >>> from sklearn.metrics import roc_auc_score

    >>> roc_auc_score(y_train_5, y_scores) 0.9611778893101814'
  sentences:
  - recall
  - encoders
  - Mini-batch Gradient Descent
- source_sentence: 'Now let’s look a bit more closely at the Keras preprocessing layers.

    Keras Preprocessing Layers

    The TensorFlow team is working on providing a set of standard Keras preprocessing
    layers. They will probably be available by the time you read this; however, the
    API may change slightly by then, so please refer to the notebook for this chapter
    if anything behaves unexpectedly. This new API will likely supersede the existing
    Feature'
  sentences:
  - convolutional layer
  - Pearson's r
  - preprocessing layers
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision 12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Now let’s look a bit more closely at the Keras preprocessing layers.\nKeras Preprocessing Layers\nThe TensorFlow team is working on providing a set of standard Keras preprocessing layers. They will probably be available by the time you read this; however, the API may change slightly by then, so please refer to the notebook for this chapter if anything behaves unexpectedly. This new API will likely supersede the existing Feature',
    'preprocessing layers',
    "Pearson's r",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 3,552 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                       | sentence_2                                                                       |
  |:--------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                           | string                                                                           |
  | details | <ul><li>min: 4 tokens</li><li>mean: 258.82 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 4.63 tokens</li><li>max: 11 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 5.46 tokens</li><li>max: 14 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | sentence_1                     | sentence_2                             |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|:---------------------------------------|
  | <code>Figure 16-6. Neural machine translation using an Encoder–Decoder network with an attention model<br>But where do these α(t,i) weights come from? It’s actually pretty simple: they are generated by a type of small neural network called an alignment model (or an attention layer), which is trained jointly with the rest of the Encoder–Decoder model. This alignment model is illustrated on the righthand side of Figure 16-6. It starts with a time-distributed Dense layer15 with a single neuron, which receives as input all the encoder outputs, concatenated with the decoder’s previous hidden state (e.g., h(2)). This layer outputs a score (or energy) for each encoder output (e.g., e(3, 2)): this score measures how well each output is aligned with the decoder’s previous hidden state. Finally, all the scores go through a softmax layer to get a final weight for each encoder output (e.g., α(3,2)). All the weights for a given decoder time step add up to 1 (since the softmax layer is not time-distributed)....</code>    | <code>softmax</code>           | <code>recurrent neural networks</code> |
  | <code>The dashed lines represent the points where the decision function is equal to 1 or –1: they are parallel and at equal distance to the decision boundary, and they form a margin around it. Training a linear SVM classifier means finding the values of w and b that make this margin as wide as possible while avoiding margin violations (hard margin) or limiting them (soft margin).<br>Training Objective<br>Consider the slope of the decision function: it is equal to the norm of the weight vector, ∥ w ∥. If we divide this slope by 2, the points where the decision function is equal to ±1 are going to be twice as far away from the decision boundary. In other words, dividing the slope by 2 will multiply the margin by 2. This may be easier to visualize in 2D, as shown in Figure 5-13. The smaller the weight vector w, the larger the margin.</code>                                                                                                                                                                              | <code>margin violations</code> | <code>meta learners</code>             |
  | <code>the word “are,” and so on. Assuming the vocabulary has 10,000 words, each model will output 10,000 probabilities.<br>Next, we compute the probabilities of each of the 30,000 two-word sentences that these models considered (3 × 10,000). We do this by multiplying the estimated conditional probability of each word by the estimated probability of the sentence it completes. For example, the estimated probability of the sentence “How” was 75%, while the estimated conditional probability of the word “will” (given that the first word is “How”) was 36%, so the estimated probability of the sentence “How will” is 75% × 36% = 27%. After computing the probabilities of all 30,000 two-word sentences, we keep only the top 3. Perhaps they all start with the word “How”: “How will” (27%), “How are” (24%), and “How do” (12%). Right now, the sentence “How will” is winning, but “How are” has not been eliminated.<br>Then we repeat the same process: we use three models to predict the next word in each of these thre...</code> | <code>TensorFlow Addons</code> | <code>Elastic Net</code>               |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 2.2523 | 500  | 4.3522        |


### Framework Versions
- Python: 3.12.9
- Sentence Transformers: 3.4.1
- Transformers: 4.50.3
- PyTorch: 2.6.0
- Accelerate: 1.6.0
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->