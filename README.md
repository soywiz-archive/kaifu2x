## Waifu2x implementation in pure Kotlin

[![Build Status](https://travis-ci.org/soywiz/kaifu2x.svg?branch=master)](https://travis-ci.org/soywiz/kaifu2x)

Waifu2x is a upscaler/noise-reductor for anime-style images based on convolutional neural networks.
Original implementation [was written in LUA](https://github.com/lltcggie/waifu2x-caffe), and there is a [very simple python based implementation](https://marcan.st/transf/waifu2x.py).
It uses a [caffee-based](http://caffe.berkeleyvision.org/) deep learning models.

Kotlin implementation uses [Korlib's Korim](https://github.com/korlibs/korim) for image processing.
And it includes code to process convulational 3x3 kernels on a float matrix.

![](/docs/kaifu2x.side2side.png)

### How to use CLI?

You can grab a precompiled jar from [github's the Releases page](https://github.com/soywiz/kaifu2x/releases/)

```
./gradlew fatJar
cd build/libs
java -jar kaifu2x-all.jar -n0 -s2 input.png output.png
```

Install kaifu2x binary in /usr/local/bin:

```
./gradlew installCli
```

### How to use CLI using kscript?

Create a file named `kaifu2x` with this contents:
```kotlin
#!/usr/bin/env kscript
//DEPS com.soywiz:kaifu2x:0.3.0
com.soywiz.kaifu2x.Kaifu2xCli.main(args)
```

Run `chmod +x kaifu2x` to give permissions.

You will need kscript:

* Using [brew](https://brew.sh/) run `brew install holgerbrandl/tap/kscript`
* Using [sdkman](http://sdkman.io/), install `sdk install kscript`

Note that first time you call the script it will take sometime, but further executions will be faster.

If you want to try it out without installing anything else than kscript or manually downloading any image:
```
brew install holgerbrandl/tap/kscript
kscript https://raw.githubusercontent.com/soywiz/kaifu2x/8e1e296bfcbb5e06f384e206ef3bb6fcb8ea3dd4/kaifu2x.kscript -s2 https://raw.githubusercontent.com/soywiz/kaifu2x/a9c863d2a181c5906f6e00726f72e93354418086/docs/goku_small_bg.png goku_small_bg.2x.png
```

### How to use as library?

It is published to maven central. In your `build.gradle` (or maven equivalent):
```
compile "com.soywiz:kaifu2x:0.3.0"
```

Exposed API:
```
package com.soywiz.kaifu2x

object Kaifu2x {
	suspend fun noiseReductionRgba(image: Bitmap32, noise: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128, output: PrintStream? = System.err): Bitmap32
	suspend fun scaleRgba(image: Bitmap32, scale: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128, output: PrintStream? = System.err): Bitmap32
}
```

### Help

```
kaifu2x - 0.3.0 - 2017

Usage: kaifu2x [switches] <input.png> <output.png>

Available switches:
  -h        - Displays this help
  -v        - Displays version
  -n[0-3]   - Noise reduction [default to 0 (no noise reduction)]
  -s[1-2]   - Scale level 1=1x, 2=2x [default to 1 (no scale)]
  -cs<X>    - Chunk size [default to 128]
  -q[0-100] - The quality of the output (JPG, PNG) [default=100]
  -mt       - Multi Threaded [default]
  -st       - Single Threaded
  -cl       - Process Luminance
  -cla      - Process Luminance & Alpha [default]
  -clca     - Process Luminance & Chroma & Alpha
  ```

### Some numbers (v0.2.0)

~~**Note:** As a performance example in a `MBP13@2.4ghz`
it takes 4 minutes to process a single component for a 470x750 image
for an output of 940x1500.~~

And memory used:
~~**Used:** 1.6GB, **Max Heap:** 2GB~~

~~Think that in the last step it has to keep 256 times (128 for the input, and 128 for the output)
the size of your uncompressed 2x image in memory.~~

~~So a 940x1500 float components, requires 5.5MB, and 256 times: 1408 MB + some extra stuff like temp buffers and so.~~

~~**NOTE:** Future versions will use less memory: https://github.com/soywiz/kaifu2x/issues/1 but that will require
fixing an issue on edges (probably padding-related).~~

### Some numbers (v0.3.0)

**Note:** As a performance example in a `MBP13@2.4ghz`
it takes 2 minutes to scale 2x a single component for a 470x750 image
for an output of 940x1500.

Version 0.3.0, successfully partition the image in chunks of 128x128 by default (you can adjust chunk size).
So the memory requirements are now much lower. 128*128*4*256=16MB, and it is typical that the cli uses around ~50MB
for any image size, though times still are slow until hardware acceleration is implemented. Also processor caches
are most likely to hit, so for bigger images this is better.

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
