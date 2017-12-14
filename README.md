## Waifu2x implementation in pure Kotlin

Waifu2x is a upscaler/noise-reductor for anime-style images based on convolutional neural networks.
Original implementation [was written in LUA](https://github.com/lltcggie/waifu2x-caffe), and there is a [very simple python based implementation](https://marcan.st/transf/waifu2x.py).
It uses a [caffee-based](http://caffe.berkeleyvision.org/) deep learning models.

Kotlin implementation uses [Korlib's Korim](https://github.com/korlibs/korim) for image processing.
And it includes code to process convulational 3x3 kernels on a float matrix.

![](/docs/kaifu2x.side2side.png)

### How to use?

You can grab a precompiled jar from [github's the Releases page](https://github.com/soywiz/kaifu2x/releases/)

```
./gradlew fatJar
cd build/libs
java -jar kaifu2x-all.jar input.png output.png
```

### Some numbers

**Note:** As a performance example in a `MBP13@2.4ghz`
it takes 4 minutes to process a single component for a 470x750 image
for an output of 940x1500.

And memory used:
**Used:** 1.6GB, **Max Heap:** 2GB

Think that in the last step it has to keep 256 times (128 for the input, and 128 for the output)
the size of your uncompressed 2x image in memory.

So a 940x1500 float components, requires 5.5MB, and 256 times: 1408 MB + some extra stuff like temp buffers and so.

### How does this work?

#### Nearest neighbour scaling

First of all, we scale the image using a nearest neighbour approach. That's it:

![](/docs/goku_small_bg.png)
![](/docs/kaifu2x.nearest.2x.png)

#### YCbCr color space

In waifu2x we have to compute images component-wise. So we cannot process RGBA at once, and we have to process
each component first.

With RGB, we have to process all three components to get a reasonable result.
But with other color spaces we can reduce the number of components we process and also improve end quality.

With [YCbCr](https://en.wikipedia.org/wiki/YCbCr) or [YUV](https://en.wikipedia.org/wiki/YUV) color spaces
we divide it in [luminance component](https://en.wikipedia.org/wiki/Luma_(video)) and [chroma components](https://en.wikipedia.org/wiki/Chrominance).
Separating it, we can just process luminance and keep chromance intact.

YCbCr decomposition representing each component as grayscale:

![](/docs/kaifu2x.YYYA.png)![](/docs/kaifu2x.CbCbCbA.png)![](/docs/kaifu2x.CrCrCrA.png)

So for waifu2x we can just use this component, to reduce times by three with a pretty acceptable result:

![](/docs/kaifu2x.YYYA.png)

Also to get good enough results we have to process alpha channel too, in the case there is alpha information on it.

#### Waifu2x input

The input of waifu2x is a pixelated 2x2 grayscale image represented as floats in the range of [0f, 1f]

### Optimizations done

#### Sliding memory reading for convolution kernel

Reduced from 30 seconds to 24 seconds in a `MBP13@2.4ghz`

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

Sadly we can't do the same for vertical, so we still are reading 3 times the required pixels, but much better than 9 times.
We cannot reuse partial results like in dynamic programming, so probably not much to optimize here except for SIMD.

#### Parallelize

Reduced from 24 seconds to 12 seconds in a `MBP13@2.4ghz`

This one is pretty straight forward: and it is to parallelize work in threads.
I have tried several places for parallelizing to reduce the overhead.
Since each of the 7 steps depends on the previous ones, that part is not parallelizable. Also it is not too big.
So I have tried to parallelize the convolution work in a per row basis. But there were too much calls to this
so the overhead was big.
In the end I placed the parallelization in a previous step.

#### Limit memory and unified implementation

At this point I unified single threaded and multithreaded implementations. I used a fixed thread pool and manually
assigned tasks so each thread just requires two additional arrays.

#### Limit allocations

In order to avoid tons of allocations, copies and so on, I preallocated all the required arrays for each step at once.
Then instead of using immutable arrays, I changed operatins to be mutable and to work on existing arrays.

#### Future optimizations

Since we can't do SIMD optimizations manually in the JVM. And there are no guarantees that the JVM uses SIMD
instructions, our only option here is to use libraries for parallelizing mathematical operations either in the CPU
and the GPU or even with shaders (GlSl for example).
That's probably out of scope for this library (at least at this point), since the aim here is to
illustrate how does this works internally and to provide a portable implementation that works out of the box on mac.
