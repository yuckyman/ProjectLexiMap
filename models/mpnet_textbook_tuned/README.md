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
- source_sentence: "Now you have all the tools and knowledge you need to create state-of-the-art\
    \ neural net architectures and train them at scale using various distribution\
    \ strategies, on your own infrastructure or on the cloud—and you can even perform\
    \ powerful Bayesian optimization to fine-tune the hyperparameters!\nExercises\n\
    1.\tWhat does a SavedModel contain? How do you inspect its content?\n2.\tWhen\
    \ should you use TF Serving? What are its main features? What are some tools you\
    \ can use to deploy it?\n3.\tHow do you deploy a model across multiple TF Serving\
    \ instances?\n4.\tWhen should you use the gRPC API rather than the REST API to\
    \ query a model served by TF Serving?\n5.\tWhat are the different ways TFLite\
    \ reduces a model’s size to make it run on a mobile or embedded device?\n6.\t\
    What is quantization-aware training, and why would you need it?\n7.\tWhat are\
    \ model parallelism and data parallelism? Why is the latter generally recommended?\n\
    8.\tWhen training a model across multiple servers, what distribution strategies\
    \ can you use? How do you choose which one to use?\n9.\tTrain a model (any model\
    \ you like) and deploy it to TF Serving or Google Cloud AI Platform. Write the\
    \ client code to query it using the REST API or the gRPC"
  sentences:
  - protobufs
  - data parallelism
  - Residual Network
- source_sentence: "This diagram also illustrates the fact that training a model means\
    \ searching for a combination of model parameters that minimizes a cost function\
    \ (over the training set). It is a search in the model’s parameter space: the\
    \ more parameters a model has, the more dimensions this space has, and the harder\
    \ the search is: searching for a needle in a 300-dimensional haystack is much\
    \ trickier than in 3 dimensions. Fortunately, since the cost function is convex\
    \ in the case of Linear Regression, the needle is simply at the bottom of the\
    \ bowl.\nBatch Gradient Descent\nTo implement Gradient Descent, you need to compute\
    \ the gradient of the cost function with regard to each model parameter θj. In\
    \ other words, you need to calculate how much the cost function will change if\
    \ you change θj just a little bit. This is called a partial derivative. It is\
    \ like asking “What is the slope of the mountain under my feet if I face east?”\
    \ and then asking the same question facing north (and so on for all other dimensions,\
    \ if you can imagine a universe with more than three dimensions). Equation 4-5\
    \ computes the partial derivative of the cost function with regard to parameter\
    \ θj, noted ∂ MSE(θ) / ∂θj.\nEquation 4-5. Partial derivatives of the cost function\n\
    \ ∂ MSE θ  =  2 ∑m  θ⊺x i − y i  x i"
  sentences:
  - partial derivatives
  - Machine Learning
  - pooling layer
- source_sentence: "Figure 1-7. An unlabeled training set for unsupervised learning\n\
    Here are some of the most important unsupervised learning algorithms (most of\
    \ these are covered in Chapters 8 and 9):\n•\tClustering\n—\tK-Means\n—\tDBSCAN\n\
    —\tHierarchical Cluster Analysis (HCA)\n•\tAnomaly detection and novelty detection\n\
    —\tOne-class SVM\n—\tIsolation Forest\n•\tVisualization and dimensionality reduction\n\
    —\tPrincipal Component Analysis (PCA)\n—\tKernel PCA\n—\tLocally Linear Embedding\
    \ (LLE)\n—\tt-Distributed Stochastic Neighbor Embedding (t-SNE)\n•\tAssociation\
    \ rule learning\n—\tApriori\n—\tEclat\nFor example, say you have a lot of data\
    \ about your blog’s visitors. You may want to run a clustering algorithm to try\
    \ to detect groups of similar visitors (Figure 1-8). At no point do you tell the\
    \ algorithm which group a visitor belongs to: it finds those connections without\
    \ your help. For example, it might notice that 40% of your visitors are males\
    \ who love comic books and generally read your blog in the evening, while 20%\
    \ are young sci-fi lovers who visit during the weekends. If you use a hierarchical\
    \ clustering algorithm, it may also subdivide each group into smaller groups.\
    \ This may help you target your posts for each group."
  sentences:
  - parameter servers
  - softmax
  - anomaly detection
- source_sentence: 'with a branch for every pair of clusters that merged, you would
    get a binary tree of clusters, where the leaves are the individual instances.
    This approach scales very well to large numbers of instances or clusters. It can
    capture clusters of various shapes, it produces a flexible and informative cluster
    tree instead of forcing you to choose a particular cluster scale, and it can be
    used with any pairwise distance. It can scale nicely to large numbers of instances
    if you provide a connectivity matrix, which is a sparse m × m matrix that indicates
    which pairs of instances are neighbors (e.g., returned by sklearn.neighbors.kneighbors_graph()).
    Without a connectivity matrix, the algorithm does not scale well to large datasets.

    BIRCH

    The BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) algorithm
    was designed specifically for very large datasets, and it can be faster than batch
    K-Means, with similar results, as long as the number of features is not too large
    (<20). During training, it builds a tree structure containing just enough information
    to quickly assign each new instance to a cluster, without having to store all
    the instances in the tree: this approach allows it to use limited memory, while
    handling huge datasets.

    Mean-Shift

    This algorithm starts by placing a circle centered on each instance; then for
    each circle it computes the mean of all the instances located within it, and it
    shifts the circle so that it is centered on the mean. Next, it iterates this mean-shifting
    step until all the circles stop moving (i.e., until each of them is centered on
    the mean of the instances it contains). Mean-Shift shifts the circles in the direction
    of higher density, until each of them has found a local density maximum. Finally,
    all the instances whose circles have settled in the same place (or close enough)
    are assigned to the same cluster. Mean-Shift has some of the same features as
    DBSCAN, like how it can find any number of clusters of any shape, it has very
    few hyperparameters (just one—the radius of the circles, called the bandwidth),
    and it relies on local density estimation. But unlike DBSCAN, Mean-Shift tends
    to chop clusters into pieces when they have internal density variations. Unfortunately,
    its computational complexity is O(m2), so it is not suited for large datasets.

    Affinity propagation

    This algorithm uses a voting system, where instances vote for similar instances
    to be their representatives, and once the algorithm converges, each representative
    and its voters form a cluster. Affinity propagation can detect any number of clusters
    of different sizes. Unfortunately, this algorithm has a computational complexity
    of O(m2), so it too is not suited for large datasets.

    Spectral clustering

    This algorithm takes a similarity matrix between the instances and creates a low-
    dimensional embedding from it (i.e., it reduces its dimensionality), then it uses'
  sentences:
  - DBSCAN
  - mAP
  - vocabulary
- source_sentence: "7.\tWhat is an off-policy RL algorithm?\n8.\tUse policy gradients\
    \ to solve OpenAI Gym’s LunarLander-v2 environment. You will need to install the\
    \ Box2D dependencies (python3 -m pip install -U gym[box2d]).\n9.\tUse TF-Agents\
    \ to train an agent that can achieve a superhuman level at SpaceInvaders-v4 using\
    \ any of the available algorithms.\n10.\tIf you have about $100 to spare, you\
    \ can purchase a Raspberry Pi 3 plus some cheap robotics components, install TensorFlow\
    \ on the Pi, and go wild! For an example, check out this fun post by Lukas Biewald,\
    \ or take a look at GoPiGo or BrickPi. Start with simple goals, like making the\
    \ robot turn around to find the brightest angle (if it has a light sensor) or\
    \ the closest object (if it has a sonar sensor), and move in that direction. Then\
    \ you can start using Deep Learning: for example, if the robot has a camera, you\
    \ can try to implement an object detection algorithm so it detects people and\
    \ moves toward them. You can also try to use RL to make the agent learn on its\
    \ own how to use the motors to achieve that goal. Have fun!\nSolutions to these\
    \ exercises are available in Appendix A."
  sentences:
  - policy gradients
  - tolerance hyperparameter
  - PER
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
    '7.\tWhat is an off-policy RL algorithm?\n8.\tUse policy gradients to solve OpenAI Gym’s LunarLander-v2 environment. You will need to install the Box2D dependencies (python3 -m pip install -U gym[box2d]).\n9.\tUse TF-Agents to train an agent that can achieve a superhuman level at SpaceInvaders-v4 using any of the available algorithms.\n10.\tIf you have about $100 to spare, you can purchase a Raspberry Pi 3 plus some cheap robotics components, install TensorFlow on the Pi, and go wild! For an example, check out this fun post by Lukas Biewald, or take a look at GoPiGo or BrickPi. Start with simple goals, like making the robot turn around to find the brightest angle (if it has a light sensor) or the closest object (if it has a sonar sensor), and move in that direction. Then you can start using Deep Learning: for example, if the robot has a camera, you can try to implement an object detection algorithm so it detects people and moves toward them. You can also try to use RL to make the agent learn on its own how to use the motors to achieve that goal. Have fun!\nSolutions to these exercises are available in Appendix A.',
    'PER',
    'tolerance hyperparameter',
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
  |         | sentence_0                                                                          | sentence_1                                                                      | sentence_2                                                                       |
  |:--------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                          | string                                                                           |
  | details | <ul><li>min: 4 tokens</li><li>mean: 260.94 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 4.6 tokens</li><li>max: 11 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 5.33 tokens</li><li>max: 14 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | sentence_1               | sentence_2                       |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|:---------------------------------|
  | <code>Figure 18-7. Example of a Markov chain<br>Suppose that the process starts in state s0, and there is a 70% chance that it will remain in that state at the next step. Eventually it is bound to leave that state and never come back because no other state points back to s0. If it goes to state s1, it will then most likely go to state s2 (90% probability), then immediately back to state s1</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>PPO</code>         | <code>SE-ResNet</code>           |
  | <code>iance, generally yielding an overall better model. The following BaggingClassifier<br>is equivalent to the previous RandomForestClassifier:<br>bag_clf = BaggingClassifier( DecisionTreeClassifier(max_features="auto", max_leaf_nodes=16), n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)<br>Extra-Trees<br>When you are growing a tree in a Random Forest, at each node only a random subset of the features is considered for splitting (as discussed earlier). It is possible to make trees even more random by also using random thresholds for each feature rather than searching for the best possible thresholds (like regular Decision Trees do).<br>A forest of such extremely random trees is called an Extremely Randomized Trees ensemble12 (or Extra-Trees for short). Once again, this technique trades more bias for a lower variance. It also makes Extra-Trees much faster to train than regular Random Forests, because finding the best possible threshold for each feature at every node is one of the most time-co...</code>             | <code>Extra-Trees</code> | <code>PCA</code>                 |
  | <code>>>> env.observation_spec()<br>BoundedArraySpec(shape=(210, 160, 3), dtype=dtype('float32'), name=None,<br>minimum=[[[0. 0. 0.], [0. 0. 0.],...]],<br>maximum=[[[255., 255., 255.], [255., 255., 255.], ...]])<br>>>> env.action_spec()<br>BoundedArraySpec(shape=(), dtype=dtype('int64'), name=None, minimum=0, maximum=3)<br>>>> env.time_step_spec()<br>TimeStep(step_type=ArraySpec(shape=(), dtype=dtype('int32'), name='step_type'), reward=ArraySpec(shape=(), dtype=dtype('float32'), name='reward'), discount=BoundedArraySpec(shape=(), ..., minimum=0.0, maximum=1.0), observation=BoundedArraySpec(shape=(210, 160, 3), ...))<br>As you can see, the observations are simply screenshots of the Atari screen, represented as NumPy arrays of shape [210, 160, 3]. To render an environment, you can call env.render(mode="human"), and if you want to get back the image in the form of a NumPy array, just call env.render(mode="rgb_array") (unlike in OpenAI Gym, this is the default mode).<br>There are four actions available. Gym’s Atari enviro...</code> | <code>IS</code>          | <code>stratified sampling</code> |
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
| 2.2523 | 500  | 4.3399        |


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