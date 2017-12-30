package com.soywiz.kaifu2x.tf

import com.soywiz.kaifu2x.getScale2xModel
import com.soywiz.kaifu2x.util.readChannelf
import com.soywiz.klock.Klock
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korim.bitmap.sliceWithSize
import com.soywiz.korim.format.PNG
import com.soywiz.korio.Korio
import com.soywiz.korio.vfs.localCurrentDirVfs
import org.tensorflow.*
import org.tensorflow.types.UInt8
import java.nio.FloatBuffer


object TensorFlowExample {
    @JvmStatic
    fun main(args: Array<String>) = Korio {
        val imageBytes = localCurrentDirVfs["src/test/resources/goku_small_bg.png"].readBytes()
        val model = getScale2xModel()

        val image32 = PNG.decode(imageBytes).toBMP32().sliceWithSize(0, 0, 16, 16).extract()
        //val image32 = PNG.decode(imageBytes).toBMP32().sliceWithSize(0, 0, 32, 32).extract()
        //val image32 = PNG.decode(imageBytes).toBMP32()
        val image = image32.readChannelf(BitmapChannel.RED).data

        TensorGraph {
            val input = image.const.reshape(1, image32.width, image32.height, 1)

            var inputs = arrayListOf(input)
            println("Generating graph...")
            val ZERO = 0f.const
            val ZERO_POINT_ONE = 1f.const

            var totalConvolutions2 = 0
            var totalConvolutions = 0
            for (step in model.steps) {
                println("" + step.nInputPlane + " -> " + step.nOutputPlane)
                val outputs = arrayListOf<TensorOutput<Float>>()
                for (wm in 0 until step.weight.size) {
                    val weights = step.weight[wm]
                    val bias = step.bias[wm]
                    val partials = arrayListOf<TensorOutput<Float>>()

                    // @TODO: We can do this at once!
                    for (n in 0 until weights.size) {
                        val input = inputs[n]
                        val weight = weights[n]
                        val kernel = weight.const.reshape(3, 3, 1, 1)
                        val output = input.conv2d(kernel)
                        partials += output
                        //return@TensorGraph

                        totalConvolutions++
                    }

                    totalConvolutions2++

                    val partial = partials.sum() + bias.const
                    val output = max(partial, ZERO) + (min(partial, ZERO) * ZERO_POINT_ONE)
                    outputs += output
                    //val res = output.fetch().getFloats()
                    //println(output.fetch().getFloats().toList())
                    //println(weightList.toList().map { it.toList() })
                }
                inputs = outputs
                //println("[1]")
                //outputs.map { it.fetch().getFloats() }
                //println("[2]")
                //break
            }
            // @TODO: We should do just 7 convolution batches instead of 31904
            println("Computing... ($totalConvolutions2, $totalConvolutions)")
            println(inputs.size)

            val start = Klock.currentTimeMillis()
            println(inputs[0].fetch().getFloats())
            val end = Klock.currentTimeMillis()
            val elapsed = end - start
            println("Done ($elapsed)")
        }

        /*
        TensorGraph {
            //val kernel = floatArrayOf(
            //        0f, 0.5f, 0f,
            //        0f, 0.5f, 0f,
            //        0f, 0f, 0f
            //).const.reshape(3, 3, 1, 1)

            val kernel = 1f.const.constFill(3, 3, 1, 1)

            //val kernel = floatArrayOf(
            //        0.1f
            //).const.reshape(1, 1, 1, 1)

            val image = imageBytes.const.decodePng(channels = 4).castToFloat() / 255f
            //val image2 = image[0..15, 0..15, 0..0].addDimension()
            //println(image.dimensions)
            val image2 = image.slice(0 until 32, 0 until 32, 0 until 4).reshape(4, 32, 32, 1)
            println(image2.out)
            //val image3 = image2.depthwiseConv2d(kernel = kernel, strides = intArrayOf(1, 1, 1, 1), padding = "VALID")
            val image3 = image2.conv2d(kernel = kernel, strides = intArrayOf(1, 1, 1, 1), padding = TensorGraph.Padding.VALID)

            val result = image3.fetch()
            println(result.dimensions.toList())
            println(result.getFloats().toList())
        }
        */

        /*
        Graph().use { g ->
            val b = GraphBuilder(g)
            val image = b.decodePng(b.constant("input", imageBytes), 4L)
            val scale255 = b.constant("scale255", 255f)
            //val imageFloat = b.div(b.cast(image, java.lang.Float::class.java), scale255)
            val imageFloat = b.div(b.cast(image, java.lang.Float::class.java) as Output<Float>, scale255)
            println(image.dataType())
            Session(g).use { s ->
                val result = s.runner().fetch(imageFloat).run().first()
                println(result)
                println(result.numDimensions())
                println(result.numElements())
                println(result.shape().toList())
                val out = FloatArray(result.numElements())
                result.writeTo(FloatBuffer.wrap(out))
                //result.writeTo()
                println(out.size)
                println(out.toList())
            }
        }
        */
    }
}

class TensorOutput<T>(val g: Graph, val out: Output<T>) {
    val dimensions by lazy { val shape = out.shape(); (0 until shape.numDimensions()).map { shape.size(it).toInt() } }
    override fun toString(): String = "$out"
}

class TensorResult<T>(val tensor: Tensor<T>) {
    val dimensions by lazy { tensor.shape() }
}

fun TensorResult<Float>.getFloats(): FloatArray {
    val fa = FloatArray(tensor.numElements())
    tensor.writeTo(FloatBuffer.wrap(fa))
    return fa
}

class TensorGraph(val g: Graph) {
    companion object {
        operator fun <T> invoke(callback: TensorGraph.() -> T): T {
            return Graph().use { g ->
                callback(TensorGraph(g))
            }
        }
    }

    fun <T> TensorOutput<T>.fetch(): TensorResult<T> {
        return Session(g).use { s ->
            TensorResult<T>(s.runner().fetch(out).run().first() as Tensor<T>)
        }
    }

    operator fun <T> TensorOutput<T>.get(vararg ranges: IntRange): TensorOutput<T> = slice(*ranges)

    fun <T> Iterable<TensorOutput<T>>.sum(): TensorOutput<T> {
        return g.opBuilder("AddN", "AddN${lastId++}")
                .addInputList(this.map { it.out }.toTypedArray())
                .build().output<T>(0).tf
    }

    fun <T> max(l: TensorOutput<T>, r: TensorOutput<T>): TensorOutput<T> = binaryOp("Maximum", l, r)
    fun <T> min(l: TensorOutput<T>, r: TensorOutput<T>): TensorOutput<T> = binaryOp("Minimum", l, r)

    fun <T> TensorOutput<T>.slice(vararg ranges: IntRange): TensorOutput<T> {
        val begin = ranges.map { it.start.toLong() }.toLongArray()
        val size = ranges.map { (it.endInclusive.toLong() - it.start.toLong()) + 1 }.toLongArray()

        //println(size.toList())

        return g.opBuilder("Slice", "Slice${lastId++}")
                .addInput(this.out)
                .addInput(begin.const.out)
                .addInput(size.const.out)
                .build().output<T>(0).tf

    }

    fun <T> TensorOutput<T>.reshape(vararg dims: Int): TensorOutput<T> {
        return g.opBuilder("Reshape", "Reshape${lastId++}")
                .addInput(this.out)
                .addInput(dims.const.out)
                .build().output<T>(0).tf
    }

    fun <T> TensorOutput<T>.depthwiseConv2d(kernel: TensorOutput<T>, strides: IntArray = intArrayOf(1, 1, 1, 1), padding: Padding = Padding.VALID): TensorOutput<T> {
        return g.opBuilder("DepthwiseConv2dNative", "DepthwiseConv2dNative${lastId++}")
                .addInput(this.out)
                .addInput(kernel.out)
                .setAttr("strides", strides.map { it.toLong() }.toLongArray())
                .setAttr("padding", padding.name)
                .build().output<T>(0).tf
    }

    enum class Padding { VALID, SAME }

    fun <T> TensorOutput<T>.conv2d(kernel: TensorOutput<T>, strides: IntArray = intArrayOf(1, 1, 1, 1), padding: Padding = Padding.VALID): TensorOutput<T> {
        return g.opBuilder("Conv2D", "Conv2D${lastId++}")
                .addInput(this.out)
                .addInput(kernel.out)
                .setAttr("strides", strides.map { it.toLong() }.toLongArray())
                .setAttr("padding", padding.name)
                .build()
                .output<T>(0)
                .tf
    }

    private val <T> Output<T>.tf get() = TensorOutput(g, this)

    private fun <T> binaryOp(type: String, in1: TensorOutput<T>, in2: TensorOutput<T>): TensorOutput<T> {
        return g.opBuilder(type, "$type${lastId++}").addInput(in1.out).addInput(in2.out).build().output<T>(0).tf
    }

    private fun <T, U, V> binaryOp3(type: String, in1: Output<U>, in2: Output<V>): Output<T> {
        return g.opBuilder(type, "$type${lastId++}").addInput(in1).addInput(in2).build().output(0)
    }

    fun <T> constant(name: String, value: Any, type: Class<T>): TensorOutput<T> {
        Tensor.create(value, type).use { t ->
            return g.opBuilder("Const", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", t)
                    .build()
                    .output<T>(0)
                    .tf
        }
    }

    private fun decodeImage(op: String, contents: TensorOutput<String>, channels: Long): TensorOutput<UInt8> {
        return g.opBuilder(op, op)
                .addInput(contents.out)
                .setAttr("channels", channels)
                .build()
                .output<UInt8>(0)
                .tf
    }

    fun <T, U> cast(value: TensorOutput<T>, type: Class<U>): TensorOutput<U> {
        val dtype = DataType.fromClass(type)
        return g.opBuilder("Cast", "Cast${lastId++}")
                .addInput(value.out)
                .setAttr("DstT", dtype)
                .build()
                .output<U>(0).tf
    }

    fun <T> TensorOutput<T>.castToFloat() = cast(this, java.lang.Float::class.java) as TensorOutput<Float>
    fun <T> resizeBilinear(images: Output<T>, size: Output<Int>): Output<Float> = binaryOp3("ResizeBilinear", images, size)
    fun <T> expandDims(input: Output<T>, dim: Output<Int>): Output<T> = binaryOp3("ExpandDims", input, dim)

    fun <T> TensorOutput<T>.addDimension(pos: Int = -1): TensorOutput<T> = expandDims(this.out, pos.const.out).tf


    fun TensorOutput<String>.decodeJpeg(channels: Number): TensorOutput<UInt8> = decodeImage("DecodeJpeg", this, channels.toLong())
    fun TensorOutput<String>.decodePng(channels: Number = 4): TensorOutput<UInt8> = decodeImage("DecodePng", this, channels.toLong())

    fun constant(name: String, value: ByteArray): TensorOutput<String> = this.constant(name, value, String::class.java)
    fun constant(name: String, value: Int): TensorOutput<Int> = this.constant(name, value, java.lang.Integer::class.java) as TensorOutput<Int>
    fun constant(name: String, value: IntArray): TensorOutput<Int> = this.constant(name, value, java.lang.Integer::class.java) as TensorOutput<Int>
    fun constant(name: String, value: FloatArray): TensorOutput<Float> = this.constant(name, value, java.lang.Float::class.java) as TensorOutput<Float>
    fun constant(name: String, value: LongArray): TensorOutput<Long> = this.constant(name, value, java.lang.Long::class.java) as TensorOutput<Long>
    fun constant(name: String, value: Float): TensorOutput<Float> = this.constant(name, value, java.lang.Float::class.java) as TensorOutput<Float>

    private var lastId = 0

    fun genName() = "temp${lastId++}"

    val Float.const: TensorOutput<Float> get() = constant(genName(), this)
    val Int.const: TensorOutput<Int> get() = constant(genName(), this)
    val IntArray.const: TensorOutput<Int> get() = constant(genName(), this)
    val FloatArray.const: TensorOutput<Float> get() = constant(genName(), this)
    val LongArray.const: TensorOutput<Long> get() = constant(genName(), this)
    val ByteArray.const: TensorOutput<String> get() = constant(genName(), this)

    fun <T> TensorOutput<T>.constFill(vararg dimensions: Int): TensorOutput<T> = g.opBuilder("Fill", genName())
            .addInput(dimensions.const.out)
            .addInput(this.out)
            .build()
            .output<T>(0).tf


    operator fun <T> TensorOutput<T>.div(that: TensorOutput<T>): TensorOutput<T> = binaryOp("Div", this, that)
    operator fun <T> TensorOutput<T>.times(that: TensorOutput<T>): TensorOutput<T> = binaryOp("Mul", this, that)
    operator fun <T> TensorOutput<T>.minus(that: TensorOutput<T>): TensorOutput<T> = binaryOp("Sub", this, that)
    operator fun <T> TensorOutput<T>.plus(that: TensorOutput<T>): TensorOutput<T> = binaryOp("Add", this, that)

    operator fun TensorOutput<Float>.div(cst: Float) = binaryOp("Div", this, cst.const)
    operator fun TensorOutput<Float>.plus(cst: Float) = binaryOp("Add", this, cst.const)
    operator fun TensorOutput<Float>.minus(cst: Float) = binaryOp("Sub", this, cst.const)
    operator fun TensorOutput<Float>.times(cst: Float) = binaryOp("Mul", this, cst.const)
}
