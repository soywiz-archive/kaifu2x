## Waifu2x implementation in pure Kotlin

Waifu2x is a upscaler/noise-reductor for anime-style images based on convolutional neural networks.
Original implementation [was written in LUA](https://github.com/lltcggie/waifu2x-caffe), and there is a [very simple python based implementation](https://marcan.st/transf/waifu2x.py).
It uses a [caffee based](http://caffe.berkeleyvision.org/) deep learning models.

![](/docs/goku_small_bg.png)

![](/docs/side2side.png)

### How does this work?

#### Nearest neighbour scaling

First of all, we scale the image using a nearest neighbour approach. That's it:

![](/docs/goku_small_bg.png)
![](/docs/kaifu2x.nearest.2x.png)

#### YCbCr color space

Waifu2x requires a [YCbCr](https://en.wikipedia.org/wiki/YCbCr) image or [YUV](https://en.wikipedia.org/wiki/YUV) image (whatever), since it just uses the [luminance component](https://en.wikipedia.org/wiki/Luminance) component.

YCbCr decomposition representing each component as grayscale:

![](/docs/kaifu2x.YYYA.png)![](/docs/kaifu2x.CbCbCbA.png)![](/docs/kaifu2x.CrCrCrA.png)

So for waifu2x we are just using this:

![](/docs/kaifu2x.YYYA.png)

NOTE: We can process each component independently, specially the alpha channel. But we are not going to do this for now.

### Optimizations done

#### Sliding memory reading for convolution kernel

The initial optimization I have done to the code is to reduce memory reading at the [convolution kernel](https://docs.gimp.org/en/plug-in-convmatrix.html).
Waifu2x model uses a convolution matrix of 3x3.
For each single component, it gather 3x3 near components, multiply them by weight matrix and sum the result.
This is a task that SIMD instructions perform very well, as well as GPUs.
But in this case I'm doing a scalar implementation in pure kotlin.
So for each element in the matrix, I read 9 contiguous elements and multiply per weights.
Weights are already locals, so potentially in registers. But what about reading?
Well, actually you are reading 9 times each component, which can be optimized using sliding windows since
we are doing this sequentially.

Consider this matrix:

```
0 1 2 3
4 5 6 7
8 9 a b
c d e f
```

If we are processing the convolution from left to right, then top to down. First we would read:

```
0 1 2
4 5 6
8 9 a
```

then

```
1 2 3
5 6 7
9 a b
```

In each contiguous step (from left to right) we have 6 values that are the same that in the previous
step, but shifted. So we only have 3 new values that we have to read from memory in each step.

#### Parallelize

This one is pretty straight forward: and it is to parallelize work in threads.
I have tried several places for parallelizing to reduce the overhead.
Since each of the 7 steps depends on the previous ones, that part is not parallelizable. Also it is not too big.
So I have tried to parallelize the convolution work in a per row basis. But there were too much calls to this
so the overhead was big.
In the end I placed the parallelization in a previous step.