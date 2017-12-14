## Waifu2x implementation in pure Kotlin

Waifu2x is a upscaler/noise-reductor for anime-style images based on convolutional neural networks.
Original implementation [was written in LUA](https://github.com/lltcggie/waifu2x-caffe), and there is a [very simple python based implementation](https://marcan.st/transf/waifu2x.py).
It uses a [caffee based](http://caffe.berkeleyvision.org/) deep learning models.

![](/docs/goku_small_bg.png)

![](/docs/side2side.png)

### Optimizations done

#### Sliding memory reading for convolution kernel

The initial optimization I have done to the code is to

#### Parallelize

This one is pretty straight forward: and it is to parallelize work in threads.
I have tried several places for parallelizing to reduce the overhead.
Since each of the 7 steps depends on the previous ones, that part is not parallelizable. Also it is not too big.
So I have tried to parallelize the convolution work in a per row basis. But there were too much calls to this
so the overhead was big.
In the end I placed the parallelization in a previous step.