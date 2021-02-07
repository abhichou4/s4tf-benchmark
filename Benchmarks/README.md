# Benchmarks 

I tested the [Dense-MNIST](https://github.com/abhichou4/s4tf-benchmark/blob/main/Models/Sources/Dense.swift) on GeForce GT 710 2GB with the following results

Swift for Tensorflow Model: 

```bash
$ swift run -c release Dense-MNIST
running Forward Pass	... done! (1892.99 ms)
running One Update step	... done! (1996.68 ms)
running Total time to train	... done! (273669.12 ms)

name                 time                std        iterations
--------------------------------------------------------------
Forward Pass	               43686.000 ns ±   1.44 %      32037
One Update step	           535329.000 ns ±   0.70 %       2607
Total time to train	 273669111446.000 ns ±   0.00 %   
```

Tensorflow Model:

```bash
name	time
------------------------
Total time to train	1775645160953.0003 ns
```
