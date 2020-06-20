# Variational Inference for Random Graph Models (VIRGMo)


## Dependencies

* **python>=3.6**
* **pytorch>=1.4.0**: https://pytorch.org
* **scipy**: https://scipy.org
* **numpy**: https://www.numpy.org

## Installation

To install, run

```bash
$ python setup.py install
```

## Source code:

- [`vi_rg`](virgmo/vi_rg.py): VI for random graphs (abstract parent class)
- [`vi_sbm`](virgmo/vi_sbm.py): VI for stochastic block models
- [`vi_graphon`](virgmo/vi_graphon.py) - VI for graphons
- [`vi_hrg`](virgmo/vi_hrg.p: - VI for hyperbolic random graphs
- [`utils`](virgmo/utils.py): util functions
- [`graph_models`](virgmo/graph_models.py): random graph models for network data generation and `EdgesDataset` for effective training
- [`distributions`](virgmo/distributions): several Pytorch implementations of non-build-in distributions (including the von Mises-Fisher and hyperspherical Uniform distributions from https://github.com/nicola-decao/s-vae-pytorch)

## Examples:

1. [SBMs on simulated data](examples/Example%201%20-%20SBMs%20on%20simulated%20data.ipynb)
2. [WSBMs on simulated data](examples/Example%202%20-%20WSBMs%20on%20simulated%20data.ipynb)
3. [SBM on simulated interbank network](examples/Example%203%20-%20SBM%20on%20simulated%20interbank%20network.ipynb)
4. [Karate club network](examples/Example%204%20-%20Karate%20club%20network.ipynb)
5. [NFL 2009 network](examples/Example%205%20-%20NFL%202009%20network.ipynb)
6. [World trade network (unweighted)](examples/Example%206a%20-%20World%20trade%20network%20(unweighted).ipynb)
7. [World trade network (weighted)](examples/Example%206b%20-%20World%20trade%20network%20(weighted).ipynb)

## Feedback
For questions and comments, feel free to contact [Iurii Mozzhorin](mailto:iurii.mozzhorin@gmail.com).

## License
MIT