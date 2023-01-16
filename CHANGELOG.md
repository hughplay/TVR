# Changelog

## TODO


- [ ] other scripts
    - [ ] preprocess
        - copy properties.json and values.json to /data/trance
        - ~~reading h5py is faster than reading raw images for training, but not much, only seconds for each epoch~~, previous code has bug
        - speed is nearly the same, I believe data reading is not the bottleneck
    - [ ] gen_rc

## Currently Working On

- [x] Demo to show datasets and Experiments
    - [x] API to serve datasets
    - [x] API to serve experiments
    - [ ] ~~fill intermediate states~~


## 2023-01-09 17:20:10

- [x] merge tvr and real tvr to DeepCodebase Framework.
    - [x] dataset
    - [x] evaluation
    - [x] pipeline
    - [x] config
- [x] test old models.
    - start: 2022-12-16 22:40:11
    - fix bug: add view_idx to inputs
- [x] dataset size ablation
- [x] reinforce ablation
    - no better validation accuracy, but train acc keeps increasing, which means overfitting
        - adamw seems to be more stable, lr=0.0005 is better for now
    - try:
      - [x] fix encoder, only tune classifier and decoder
        - better than previous tunning, but still not better than baseline
      - [x] rnn dropout
        - 0.3 is best
      - [x] classifier dropout
        - 0.3 is best
- [x] test half precision
    - no speed up, weird
- [x] add transformer based model
- change primary models from concat to subtract
    - [x] reinforce ablation
    - [x] scaling data ablation

Issues:

- [x] on node 12, `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` when running `python core.py train config/event.ConcatResNet.yaml`.
    - move the whole code into DeepCodebase and run in docker.
- [x] training is slow.
    - the training stops on 4th iteration in 1st epoch.
        - seems to be caused by profiler
    - find the bottleneck through profiler
        - learn profilers
            - pytorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
            - open json files in chrome://tracing
            - nvprof: https://docs.nvidia.com/cuda/profiler-users-guide/index.html
        - the bottleneck is shown to be ReasonCriterion
            - tried to show exact modules in the criterion, but failed
            - but it should be `EventEvaluator`
                - evaluator open: 4.4it/s
                - evaluator close: 10.2it/s
        - solution: do not evaluate during training
- [x] `obj_acc` appears in the view.
    - remove old `ViewRecorder` and rename `MultiViewRecorder` to `ViewRecorder`.
