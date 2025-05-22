# vLLM Cache Eviction Simulator

## Supported Strategies
- [x] LRU
- [x] LFU
- [x] DBL (`OrderedDict` and `Priority Queue` Implementation)
- [x] DBL (with Ghost Queue)
- [x] ARC (`OrderedDict` and `Priority Queue` Implementation)
- [x] ARC (Sequence Based Evction Access Pattern, i.e. `ARCTimestampCache`)

## Simulation
### 1, Get Trace
The trace is either from the trace logger from [my vLLM Eviction Strategy Integration](https://github.com/Y-aang/vllm.git) (build vLLM from source, reference to vLLM official manual), or generated (e.g, `workload_generator.py`).

An example for its format:
```
7298710589679532017 7969495617363612671 ...
-5174525100055271610 -1796965167165991364 -172559708037438650 ...
... 

```
### 2, Simulate
First, provide Texts'/Documents' Content_hash to sample from.

#### Power-Law Distribution
`full_power_law.sh` & `full_power_law.py`: assign a global power-law distribution over all available documents and sample from it. Feed them to different eviction strategies and calculate the hit rate.

```
bash full_power_law.sh
# Or call the python script directly
python full_power_law.py --cache_size_fraction 0.02 --sequence_length 7000 --alpha 1.0
```

#### Power-Law-Based HotSpot
`local_power_law.py`: assign a global power-law distribution and randomly choose some hotspot in each window of time.

```
python local_power_law.py --cache_size_fraction 0.02 --sequence_length 7000 --alpha 1.0
```


#### Distribution Shift
`distribution_shift.py`: the docs are shuffle in each window of time and sample from a power-law distribution.

```
python distribution_shift.py --cache_size_fraction 0.02 --sequence_length 7000 --alpha 1.0
```


## Verification
Paste the content_hash logger into `vLLM_valid.txt`.

Remember to set the average length of the prompt.
```
python vLLM_validation.py --cp_ratio 6 > ./result/arc_output.txt 2>&1
```
The result should be close to the vLLM's hit rate.

## Ploting
`view_graph.ipynb`