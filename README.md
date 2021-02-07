# Swift for TensorFlow Benchmarks

Benchmarking S4TF models against Keras+TF and PyTorch models. 

For more information on the tools used for development, visit [tensorflow/swift](https://github.com/tensorflow/swift/blob/main/Installation.md#releases). Swift models are benchmarked using [google/swift-benchmark](https://github.com/google/swift-benchmark) 

## Benchmarks

To run swift benchmarks for a model on your local machine.

```bash
$ cd Benchmarks/
$ swift run -c release Dense-MNIST --help

```
## Status
- [x] Keras+TF and PyTorch Models
- [ ] Containerize Benchmarks
