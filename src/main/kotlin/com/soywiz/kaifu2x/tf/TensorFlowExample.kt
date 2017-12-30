package com.soywiz.kaifu2x.tf

import com.soywiz.kaifu2x.getScale2xModel
import com.soywiz.kaifu2x.util.readChannelf
import com.soywiz.klock.Klock
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korim.bitmap.sliceWithSize
import com.soywiz.korim.format.PNG
import com.soywiz.korio.Korio
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.vfs.localCurrentDirVfs
import org.tensorflow.*
import org.tensorflow.types.UInt8
import java.nio.FloatBuffer

object TensorFlowExample2 {
    @JvmStatic
    fun main(args: Array<String>) = Korio {
        fun packKernels(vararg kernels: FloatArray): FloatArray {
            val out = FloatArray(kernels.map { it.size }.sum())
            var z = 0
            for (y in 0 until 3) {
                for (x in 0 until 3) {
                    for (k in 0 until kernels.size) {
                        out[z++] = kernels[k][y * 3 + x]
                    }
                }
            }
            return out
        }

        TensorGraph {
            val image = floatArrayOf(
                    0f, 1f, 2f, 3f,
                    4f, 5f, 6f, 7f,
                    8f, 9f, 10f, 11f,
                    12f, 13f, 14f, 14f
            ).const.reshape(1, 4, 4, 1)
            //.reshape(2, 2).padZero(1, 1, 1, 1).reshape(1, 6, 6, 1)

            val im2 = floatArrayOf(
                    0f, 1f, 2f, 3f,
                    4f, 5f, 6f, 7f,
                    8f, 9f, 10f, 11f,
                    12f, 13f, 14f, 14f
            ).const.reshape(4, 4).padZero(1 to 1, 1 to 1)

            println(im2.fetch().getFloats().toList())


            val kernels = packKernels(
                    floatArrayOf(
                            0f, 0f, 0f,
                            0f, 1f, 0f,
                            0f, 0f, 0f
                    ),
                    floatArrayOf(
                            0f, 1f, 0f,
                            0f, 0f, 0f,
                            0f, 0f, 0f
                    ),
                    floatArrayOf(
                            0f, 0f, 0f,
                            0f, 0f, 0f,
                            0f, 1f, 0f
                    ),
                    floatArrayOf(
                            0f, 0f, 0f,
                            1f, 0f, 0f,
                            0f, 0f, 0f
                    )
            ).const.reshape(3, 3, 1, 4)

            val res = image.conv2d(kernels)

            println(res.fetch().tensor)
            println(res.fetch().getFloats().toList())
        }
    }
}


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
    override fun toString(): String = "$tensor"
}

fun TensorResult<Float>.getFloats(): FloatArray {
    val fa = FloatArray(tensor.numElements())
    tensor.writeTo(FloatBuffer.wrap(fa))
    return fa
}

class TensorGraph(val g: Graph) {
    private var lastId = 0
    fun genName() = "temp${lastId++}"

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

    private fun <T> build(type: String, vararg inputs: Any, attrs: Map<String, Any> = mapOf(), name: String = "$type${lastId++}"): TensorOutput<T> {
        val b = g.opBuilder(type, name)
        for (i in inputs) {
            when (i) {
                is TensorOutput<*> -> b.addInput(i.out)
                is Iterable<*> -> b.addInputList(i.map { (it as TensorOutput<*>).out }.toTypedArray())
                is IntArray -> b.addInput(i.const.out)
                else -> invalidOp("Unsupported $i")
            }
        }
        if (attrs.isNotEmpty()) {
            for ((k, v) in attrs) {
                when (v) {
                    is DataType -> b.setAttr(k, v)
                    is String -> b.setAttr(k, v)
                    is Boolean -> b.setAttr(k, v)
                    is Int -> b.setAttr(k, v.toLong())
                    is Long -> b.setAttr(k, v.toLong())
                    is Float -> b.setAttr(k, v)
                    is Double -> b.setAttr(k, v.toFloat())
                    is ByteArray -> b.setAttr(k, v)
                    is FloatArray -> b.setAttr(k, v)
                    is IntArray -> b.setAttr(k, v.map { it.toLong() }.toLongArray())
                    is LongArray -> b.setAttr(k, v)
                    is Tensor<*> -> b.setAttr(k, v)
                    else -> TODO("Unsupported $v ${v::class.java}")
                }
            }
        }
        return b.build().output<T>(0).tf
    }

    fun <T> Iterable<TensorOutput<T>>.sum(): TensorOutput<T> = build("AddN", this)
    fun <T> max(l: TensorOutput<T>, r: TensorOutput<T>): TensorOutput<T> = binaryOp("Maximum", l, r)
    fun <T> min(l: TensorOutput<T>, r: TensorOutput<T>): TensorOutput<T> = binaryOp("Minimum", l, r)

    fun <T> TensorOutput<T>.slice(vararg ranges: IntRange): TensorOutput<T> {
        val begin = ranges.map { it.start.toLong() }.toLongArray()
        val size = ranges.map { (it.endInclusive.toLong() - it.start.toLong()) + 1 }.toLongArray()
        return build("Slice", this, begin.const, size.const)
    }

    //fun <T> TensorOutput<T>.padZero(vararg paddings: Int): TensorOutput<T> = build("Pad", this, paddings.const)
    fun <T> TensorOutput<T>.padZero(paddings: TensorOutput<Int>): TensorOutput<T> = build("Pad", this, paddings)
    fun <T> TensorOutput<T>.padZero(vararg paddings: Pair<Int, Int>): TensorOutput<T> = build("Pad", this, paddings.flatMap { listOf(it.first, it.second) }.toIntArray().const.reshape(paddings.size, 2))
    fun <T> TensorOutput<T>.padConstant(paddings: TensorOutput<Int>, constant: TensorOutput<T> = 0.const.castTo(this.out.dataType())): TensorOutput<T> = build("PadV2", this, paddings, constant)
    fun <T> TensorOutput<T>.padMirror(paddings: TensorOutput<Int>): TensorOutput<T> = build("MirrorPad", this, paddings)
    fun <T> TensorOutput<T>.reshape(vararg dims: Int): TensorOutput<T> = build("Reshape", this, dims.const)

    fun <T> TensorOutput<T>.depthwiseConv2d(kernel: TensorOutput<T>, strides: IntArray = intArrayOf(1, 1, 1, 1), padding: Padding = Padding.VALID): TensorOutput<T> {
        return build("DepthwiseConv2dNative", this, kernel, attrs = mapOf("strides" to strides, "padding" to padding))
    }

    enum class Padding { VALID, SAME }

    fun <T> TensorOutput<T>.conv2d(kernel: TensorOutput<T>, strides: IntArray = intArrayOf(1, 1, 1, 1), padding: Padding = Padding.VALID): TensorOutput<T> {
        return build("Conv2D", this, kernel, attrs = mapOf("strides" to strides, "padding" to padding.name))
    }

    private val <T> Output<T>.tf get() = TensorOutput(g, this)

    private fun <T> binaryOp(type: String, in1: TensorOutput<T>, in2: TensorOutput<T>): TensorOutput<T> = build(type, in1, in2)
    private fun <T, U, V> binaryOp3(type: String, in1: TensorOutput<U>, in2: TensorOutput<V>): TensorOutput<T> = build<T>(type, in1, in2)
    fun <T> constant(name: String, value: Any, type: Class<T>): TensorOutput<T> {
        return Tensor.create(value, type).use { t ->
            build("Const", attrs = mapOf("dtype" to DataType.fromClass(type), "value" to t), name = name)
        }
    }

    private fun decodeImage(op: String, contents: TensorOutput<String>, channels: Long): TensorOutput<UInt8> {
        return build(op, contents, attrs = mapOf("channels" to channels))
    }

    fun <T, U> cast(value: TensorOutput<T>, dtype: DataType): TensorOutput<U> = build("Cast", value, attrs = mapOf("DstT" to dtype))
    fun <T, U> cast(value: TensorOutput<T>, type: Class<U>): TensorOutput<U> = cast(value, DataType.fromClass(type))

    fun <T> TensorOutput<T>.castToFloat() = cast(this, java.lang.Float::class.java) as TensorOutput<Float>
    fun <T, U> TensorOutput<T>.castTo(dataType: DataType): TensorOutput<U> = cast(this, dataType)
    fun <T> resizeBilinear(images: TensorOutput<T>, size: TensorOutput<Int>): TensorOutput<Float> = binaryOp3("ResizeBilinear", images, size)
    fun <T> expandDims(input: TensorOutput<T>, dim: TensorOutput<Int>): TensorOutput<T> = binaryOp3("ExpandDims", input, dim)

    fun <T> TensorOutput<T>.addDimension(pos: Int = -1): TensorOutput<T> = expandDims(this, pos.const)


    fun TensorOutput<String>.decodeJpeg(channels: Number): TensorOutput<UInt8> = decodeImage("DecodeJpeg", this, channels.toLong())
    fun TensorOutput<String>.decodePng(channels: Number = 4): TensorOutput<UInt8> = decodeImage("DecodePng", this, channels.toLong())

    fun constant(name: String, value: ByteArray): TensorOutput<String> = this.constant(name, value, String::class.java)
    fun constant(name: String, value: Int): TensorOutput<Int> = this.constant(name, value, java.lang.Integer::class.java) as TensorOutput<Int>
    fun constant(name: String, value: IntArray): TensorOutput<Int> = this.constant(name, value, java.lang.Integer::class.java) as TensorOutput<Int>
    fun constant(name: String, value: FloatArray): TensorOutput<Float> = this.constant(name, value, java.lang.Float::class.java) as TensorOutput<Float>
    fun constant(name: String, value: LongArray): TensorOutput<Long> = this.constant(name, value, java.lang.Long::class.java) as TensorOutput<Long>
    fun constant(name: String, value: Float): TensorOutput<Float> = this.constant(name, value, java.lang.Float::class.java) as TensorOutput<Float>

    val Float.const: TensorOutput<Float> get() = constant(genName(), this)
    val Int.const: TensorOutput<Int> get() = constant(genName(), this)
    val IntArray.const: TensorOutput<Int> get() = constant(genName(), this)
    val FloatArray.const: TensorOutput<Float> get() = constant(genName(), this)
    val LongArray.const: TensorOutput<Long> get() = constant(genName(), this)
    val ByteArray.const: TensorOutput<String> get() = constant(genName(), this)
    val Any.const: TensorOutput<*>
        get() = when (this) {
            is Int -> this.const
            is Float -> this.const
            is IntArray -> this.const
            is FloatArray -> this.const
            is LongArray -> this.const
            is ByteArray -> this.const
            else -> TODO("Unsupported $this")
        }

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
