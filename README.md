# The harsher critisim against Interaction Combinators and HVM

People want good stuff, so do I. TBH, I kind of consider myself a fanatic for IC replacing CUDA if this is possiible. And I think many people can relate to me, having written CUDA code in real life and finding them hard to grasp.

A sweet dream is promised but it's no where near completion, and I don't see the future yet, as many others, dispite engineers from HoC being confident what IC will be. 
In this article, I will summarize from reviews that HVM2 received, and try to make it clear what HVM2 actually is and the limitations we have for now. I hope this will get you a more cool-minded view.

## Performance

Here is some benchmarks, I gathered from the internet from different individuals.

### Benchmarks on recuring inside a perfectly balanced binary tree

From [hacker news](https://news.ycombinator.com/item?id=40392233). 
```python
def sum(depth, x):
  if depth == 0:
    return x
  else:
    fst = sum(depth-1, x*2+0) # adds the fst half
    snd = sum(depth-1, x*2+1) # adds the snd half
    return fst + snd

print(sum(30, 0))
```

| Configuration | Number of thread | Runtime | 
| - | - | - |
| Python, PyPy3 | 1 | 4.478s |
| Python, CPython 3.12 | 1 | 1min42.148s |
| Bend, CPU, Apple M3 Max | 1 | 3.5min |
| Bend, CPU, Apple M3 Max | 16 | 10.26s |
| Bend, GPU, NVIDIA RTX 4090 | 32k | 1.88s |

### energy-efficiency
- No guarantee on power efficiency at all.

## Limitations
### Limitations of the implementation(HVM2)

Some reference from [hacker news](https://news.ycombinator.com/item?id=40390287)
- No serious examples that directly compare Bend/HVM to  it's OMP/CUDA.
- Too many allocations
- No TCO
- 24bit tagged numbers
- No array data types
- Maximum 4GB heap of nodes
  - Underwhelming, as workloads that actually would want to run on GPU may bypass such restriction.
  - Scaling is pitched, but with less than 4GB memory for data, scaling is unlikely to kick in and beat performant implementation that scales worse.

### Limitation on the theretical point of view

Some reference from  [hacker news](https://news.ycombinator.com/item?id=40394814)

- Graph Traversal simply doesn't map well on hardware.
- Premise of optimal reduction is valid, yet kernels need to be written in a parallelizable way(i.e. no data dependency, no recursion).
- In real world HPC, tree-like structures are non-existent. Arrays are king.

## Expectation not fullfilled
Some reference from [reddit r/CUDA](https://www.reddit.com/r/CUDA/comments/1cu5oce/comment/l4gngzc/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)
- "How fast would a naive conv or matmul run compared to sota on this? If it just schedules loops on the gpu im not amazed"
- We care scaling only because we care the performance it might imply.
- How is it not a solution in search for a problem? Where's the niche?
  - Obviously not comparable with CUDA, OpenCL, etc. As auto-parallelization could never beat hand-rolled parallel algorithm.
  - Didn't beat something like mojo or cupy as well. Don't see a promising future as well.


## Insights
- Modern GPU, while usable for Interaction Combinator reduction, is inheretly an inefficient solution for it. We may need specific hardware and architecture for Interaction Combinators.
