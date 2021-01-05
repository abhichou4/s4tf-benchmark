# Benchmarks 

I tested the [Dense-MNIST](https://github.com/abhichou4/s4tf-benchmark/blob/main/Models/Sources/Dense.swift) on GeForce GT 710 2GB with the following results

```bash
$ swift run -c release Dense-MNIST
running Loading MNIST Dataset	... done! (5500.14 ms)
running Forward Pass	... done! (2373.26 ms)
running Forward and Backward Pass (Gradients)	... done! (2121.02 ms)
running Update Weights	... done! (1970.30 ms)
running Forward Pass	... done! (2373.99 ms)
running Forward and Backward Pass (Gradients)	... done! (2182.31 ms)
running Update Weights	... done! (1969.97 ms)

name                                   time              std        iterations
------------------------------------------------------------------------------
Loading MNIST Dataset	                 1833779711.000 ns ±   0.05 %          2
Forward Pass	                               88288.000 ns ±   5.49 %      15846
Forward and Backward Pass (Gradients)	     605011.000 ns ±  32.08 %       2146
Update Weights	                            422223.000 ns ±  16.18 %       3443
Forward Pass	                               88298.000 ns ±   6.38 %      15859
Forward and Backward Pass (Gradients)	     655091.000 ns ±  22.62 %       2018
Update Weights	                            451759.000 ns ±  14.43 %       3337

```

The idea was to run benchmarks on both CPU and GPU, but it seems to always utilise GPU, even with the right [context](https://github.com/abhichou4/s4tf-benchmark/blob/a107a5b5b9360421bad5bf151edd9eedbf47f2ad/Benchmarks/Dense-MNIST/main.swift#L66-L83). I shall fix this soon.
