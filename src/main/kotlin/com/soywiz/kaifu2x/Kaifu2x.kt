package com.soywiz.kaifu2x

import com.jogamp.opencl.*
import com.soywiz.kaifu2x.util.*
import com.soywiz.klock.*
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.color.*
import com.soywiz.korim.format.*
import com.soywiz.korio.*
import com.soywiz.korio.error.*
import com.soywiz.korio.lang.*
import com.soywiz.korio.serialization.json.*
import com.soywiz.korio.util.*
import com.soywiz.korio.vfs.*
import java.io.*
import java.io.Closeable
import java.nio.*
import java.util.LinkedList
import kotlin.collections.*
import kotlin.math.*
import kotlin.system.*

fun main(args: Array<String>) = Kaifu2xCli.main(args)

object Kaifu2xCli {
    fun help() {
        System.err.println("kaifu2x - $KAIFU2X_VERSION - 2017")
        System.err.println("")
        System.err.println("Usage: kaifu2x [switches] <input.png> <output.png>")
        System.err.println("")
        System.err.println("Available switches:")
        System.err.println("  -h        - Displays this help")
        System.err.println("  -v        - Displays version")
        System.err.println("  -n[0-3]   - Noise reduction [default to 0 (no noise reduction)]")
        System.err.println("  -s[1-2]   - Scale level 1=1x, 2=2x [default to 1 (no scale)]")
        System.err.println("  -cs<X>    - Chunk size [default to 128]")
        System.err.println("  -q[0-100] - The quality of the output (JPG, PNG) [default=100]")
        System.err.println("  -cl       - Process Luminance")
        System.err.println("  -cla      - Process Luminance & Alpha")
        System.err.println("  -clca     - Process Luminance & Chroma & Alpha [default]")
    }

    fun helpAndExit(code: Int = -1) = run { help(); System.exit(code) }

    @JvmStatic
    fun main(args: Array<String>) = Korio {
        var channels = BitmapChannel.ALL.toList()
        var inputName: String? = null
        var outputName: String? = null
        var noiseReduction = 0
        var scale = 1
        var quality = 100
        var chunkSize = 128

        if (args.isEmpty()) helpAndExit()

        val argsR = LinkedList(args.toList())
        while (argsR.isNotEmpty()) {
            val c = argsR.removeFirst()
            when {
                c == "-h" -> helpAndExit()
                c == "-v" -> run { println(KAIFU2X_VERSION); exitProcess(-1) }
                c == "-cl" -> channels = listOf(BitmapChannel.Y)
                c == "-cla" -> channels = listOf(BitmapChannel.Y, BitmapChannel.A)
                c == "-clca" -> channels = BitmapChannel.ALL.toList()
                c.startsWith("-cs") -> chunkSize = c.substr(3).toIntOrNull() ?: 128
                c.startsWith("-n") -> noiseReduction = c.substr(2).toIntOrNull() ?: 0
                c.startsWith("-s") -> scale = c.substr(2).toIntOrNull() ?: 1
                c.startsWith("-q") -> quality = c.substr(2).toIntOrNull() ?: 100
                else -> {
                    if (c.startsWith("-")) invalidOp("Unknown switch $c")
                    when {
                        inputName == null -> inputName = c
                        outputName == null -> outputName = c
                        else -> invalidOp("Unexpected argument $c")
                    }
                }
            }
        }

        if (noiseReduction !in 0..3) invalidOp("nouseReduction must be between 0..3")
        if (scale !in 1..2) invalidOp("scale must be between 1..2")
        val inputFileName = inputName ?: invalidOp("Missing input file name")
        val outputFileName = outputName ?: invalidOp("Missing output file name")

        val outputExtension = PathInfo(outputFileName).extensionLC

        if (outputExtension !in listOf(
                "png",
                "jpg"
            )
        ) invalidOp("Just supported 'png' or 'jpg' outputs but found extension $outputExtension")

        defaultImageFormats.registerStandard()
        System.err.print("Reading $inputFileName...")
        val image = UniversalVfs(inputFileName).readBitmapNoNative().toBMP32()
        System.err.println("Ok")

        val noiseReductedImage =
            Kaifu2x.noiseReductionRgba(image, noiseReduction, channels, chunkSize = chunkSize)
        val scaledImage = Kaifu2x.scaleRgba(noiseReductedImage, scale, channels)

        val outFile = UniversalVfs(outputFileName).ensureParents()
        System.err.print("Writting $outputFileName...")
        scaledImage.writeTo(outFile, ImageEncodingProps(quality = quality.toDouble() / 100.0))
        System.err.println("Ok")

        if (noiseReduction == 0 && scale == 1) {
            System.err.println("WARNING!!: No operation done! Please add -nX or -sX switches to control noise reduction and scaling")
        }
    }
}

class Kaifu2xOpencl(private val model: Model, val chunkSize: Int = 128) : Closeable {
    private val steps = model.steps
    //private val ctx = CLContext.create(CLDevice.Type.GPU)
    private val ctx = try {
        CLContext.create(CLDevice.Type.GPU)
    } catch (e: Throwable) {
        CLContext.create()
    }
    val noutputs = listOf(32, 32, 64, 64, 128, 128, 1)
    private val program = ctx.createProgram(
        """
        #define hload(array, n) vload_half(n, array)
        #define hstore(array, n, value) vstore_half(value, n, array)

        #define fload(array, n) array[n]
        #define fstore(array, n, value) array[n] = value

        __kernel void waifu2x(
            const int width,
            const int height,
            const int inputs,
            __read_only global const half *krn,
            __read_only global const half *bias,
            __read_only global const float *in,
            __write_only global float *out,
            int TW, int TH, int TD
        ) {
            int area = width * height;
            int x = get_global_id(0);
            int y = get_global_id(1);
            int z = get_global_id(2);

            float acc = 0;
            int xy = x + (y * width);

            for (int i = 0; i < inputs; i++) {
                int off_kn = ((z * inputs * 3) + (i * 3));
                int off_in = (xy + (i * area));

                float3 kk[3] = {
                    vload_half3(off_kn + 0, krn),
                    vload_half3(off_kn + 1, krn),
                    vload_half3(off_kn + 2, krn)
                };

                float3 ff[3] = {
                    { in[off_in + (width * 0) + 0], in[off_in + (width * 0) + 1], in[off_in + (width * 0) + 2] },
                    { in[off_in + (width * 1) + 0], in[off_in + (width * 1) + 1], in[off_in + (width * 1) + 2] },
                    { in[off_in + (width * 2) + 0], in[off_in + (width * 2) + 1], in[off_in + (width * 2) + 2] }
                };

                float3 rr[3] = { kk[0] * ff[0], kk[1] * ff[1], kk[2] * ff[2] };

                for (int j = 0; j < 3; j++) {
                    acc += rr[j].x;
                    acc += rr[j].y;
                    acc += rr[j].z;
                }
            }
            float racc = acc + hload(bias, z);
            fstore(out, xy + (z * area), racc - 0.9 * fmin(racc, 0));
        }

        __kernel void float_to_half(
            __read_only global const float *in,
            __write_only global half *out
        ) {
            int n = get_global_id(0);
            hstore(out, n, fload(in, n));
        }
    """
    ).apply {
        build()
    }

    val waifu2xKernel = program.createCLKernel("waifu2x")
    val float_to_halfKernel = program.createCLKernel("float_to_half")


    val device = ctx.maxFlopsDevice
    val _queue: CLCommandQueue = device.createCommandQueue()

    fun float_to_half(inp: CLBuffer<*>, outp: CLBuffer<*>, count: Int) {
        float_to_halfKernel.rewind().putArg(inp).putArg(outp)
        _queue.put1DRangeKernel(float_to_halfKernel, 0L, count.toLong(), 0L)
    }

    fun float_to_half(data: FloatBuffer): ByteBuffer {
        //val len = data.limit()
        //val bo = ByteBuffer.allocateDirect(len * 2).order(ByteOrder.nativeOrder())
        //val bo2 = bo.asCharBuffer()
        //for (n in 0 until len) bo2.put(n, HalfFloat.fromFloat(data[n]).toChar())

        val len = data.limit()
        val bo = ByteBuffer.allocateDirect(len * 2).order(ByteOrder.nativeOrder())
        val bin = ctx.createBuffer(data, CLMemory.Mem.READ_ONLY)
        val bout = ctx.createBuffer(bo, CLMemory.Mem.WRITE_ONLY)

        _queue.putWriteBuffer(bin, false)
        float_to_half(bin, bout, len)
        _queue.putReadBuffer(bout, true)
        //println(bo.position())

        bout.release()
        bin.release()


        return bo
    }

    fun FloatBuffer.toHalf() = float_to_half(this)

    private val biasPerStep = model.steps.map {
        ctx.createBuffer(it.bias.check("bias").toDirectBuffer().toHalf(), CLMemory.Mem.READ_ONLY)
            .apply { _queue.putWriteBuffer(this, false) }
    }
    private val kernelPerSteps = model.steps.map {
        ctx.createBuffer(
            it.weight.flatMap { it.flatMap { it.toList() } }.toFloatArray().check("kernel").toDirectBuffer().toHalf(),
            CLMemory.Mem.READ_ONLY
        ).apply { _queue.putWriteBuffer(this, false) }
    }

    private fun FloatArray.check(name: String): FloatArray {
        val min = this.min()
        val max = this.max()
        //println("$name: $min, $max --> ${this.size}")
        return this
    }

    init {
        _queue.finish()
    }

    val inputBuffers: List<CLBuffer<FloatBuffer>> =
        (0 until 4).map { ctx.createBuffer(DirectFloatBuffer(chunkSize * chunkSize), CLMemory.Mem.READ_ONLY) }
    val inputBuffer: CLBuffer<FloatBuffer> = inputBuffers[0]

    inner class WaifuWork {
        val queue: CLCommandQueue = device.createCommandQueue()
        val obuffers =
            noutputs.map { ctx.createBuffer(DirectFloatBuffer(chunkSize * chunkSize * it), CLMemory.Mem.READ_WRITE) }
    }

    //val works = (0 until 4).map { WaifuWork() }
    //val work = works.first()
    val work = WaifuWork()

    private fun waifu2x(
        work: WaifuWork,
        i: Int,
        bw: Int,
        bh: Int,
        n_in: Int,
        n_out: Int,
        ninput: CLBuffer<*>,
        kern_buf: CLBuffer<*>,
        bias_buf: CLBuffer<*>,
        noutput: CLBuffer<*>
    ) {
        //println("i=$i, n_out=$n_out, n_in=$n_in, bw=$bw, bh=$bh, bias_buf=${bias_buf.length}, kern_buf=${kern_buf.length}")

        val TW = (bw - (2 * i)).toLong()
        val TH = (bh - (2 * i)).toLong()
        val TD = n_out.toLong()

        waifu2xKernel.rewind()
            .putArg(bw).putArg(bh)
            .putArg(n_in)
            .putArg(kern_buf).putArg(bias_buf)
            .putArg(ninput).putArg(noutput)
            .putArg(TW).putArg(TH).putArg(TD)

        work.queue.put3DRangeKernel(
            waifu2xKernel,
            0L, 0L, 0L,
            TW, TH, TD,
            0L, 0L, 0L
            //TW, TH, TD
        )
    }


    fun waifu2x(work: WaifuWork, bw: Int, bh: Int, inputBuffer: CLBuffer<FloatBuffer>) {

        var ninput: CLBuffer<FloatBuffer> = inputBuffer

        lateinit var noutput: CLBuffer<FloatBuffer>

        for (i in 0 until steps.size) {
            val n_in = steps[i].nInputPlane
            val n_out = steps[i].nOutputPlane
            val bias_buf = biasPerStep[i]
            val kern_buf = kernelPerSteps[i]

            noutput = work.obuffers[i]
            waifu2x(work, i, bw, bh, n_in, n_out, ninput, kern_buf, bias_buf, noutput)
            ninput = noutput
        }

        work.queue.putCopyBuffer(noutput, inputBuffer)
    }

    // data already have padding
    fun waifu2x(work: WaifuWork, data: FloatArray2, buffer: CLBuffer<FloatBuffer> = inputBuffer): () -> FloatArray2 {
        val pad = model.padding
        val bw = data.width
        val bh = data.height
        buffer.buffer.clear()
        buffer.buffer.put(data.data)
        buffer.buffer.flip()
        work.queue.putWriteBuffer(buffer, false)

        waifu2x(work, bw, bh, buffer)

        return {
            work.queue.putReadBuffer(buffer, true)
            val out = FloatArray(buffer.buffer.limit())
            buffer.buffer.position(0)
            buffer.buffer.get(out)
            FloatArray2(bw, bh, out)[0 until (bw - pad * 2), 0 until (bh - pad * 2)]
        }
    }

    fun waifu2x(
        data: Bitmap32,
        vararg channels: BitmapChannel = arrayOf(
            BitmapChannel.RED,
            BitmapChannel.GREEN,
            BitmapChannel.BLUE,
            BitmapChannel.ALPHA
        )
    ): Bitmap32 {
        val out = data[7 until data.width - 7, 7 until data.height - 7]

        val ochannels = arrayListOf<Pair<BitmapChannel, () -> FloatArray2>>()
        for ((index, channel) in channels.withIndex()) {
            val i = data.readChannelf(channel)
            val first = i.data[0]
            if (!i.data.all { it == first }) {
                //val gen = waifu2x(works[index], i, inputBuffers[index])
                val gen = waifu2x(work, i, inputBuffers[index])
                ochannels += channel to gen
            }
        }
        for ((channel, gen) in ochannels) {
            out.writeChannelf(channel, gen())
        }
        return out
    }

    fun waifu2xChunkedYCbCr(
        bmpYCbCr: Bitmap32,
        vararg channels: BitmapChannel = arrayOf(
            BitmapChannel.RED,
            BitmapChannel.GREEN,
            BitmapChannel.BLUE,
            BitmapChannel.ALPHA
        ),
        progress: (current: Int, total: Int) -> Unit = { current, total -> }
    ): Bitmap32 {
        val pad1 = steps.size
        val pad2 = pad1 * 2
        val pad = pad1 * 2
        //val bmpPad = Bitmap32(bmpYCbCr.width.nextAlignedTo(chunkSize) + pad2, bmpYCbCr.height.nextAlignedTo(chunkSize) + pad2)
        val bmpPad = Bitmap32(bmpYCbCr.width + pad2, bmpYCbCr.height + pad2)
        bmpPad.put(bmpYCbCr, pad1, pad1)
        val bmpOut = Bitmap32(bmpPad.width - pad2, bmpPad.height - pad2)
        val chunkSizePad = chunkSize - pad

        var currentPixels = 0
        val totalPixels = bmpOut.area

        for (cy in 0 until (bmpPad.height.toDouble() / chunkSizePad.toDouble()).toIntCeil()) {
            for (cx in 0 until (bmpPad.width.toDouble() / chunkSizePad.toDouble()).toIntCeil()) {
                val px = cx * chunkSizePad
                val py = cy * chunkSizePad
                val width = min(chunkSize, bmpPad.width - px)
                val height = min(chunkSize, bmpPad.height - py)
                //println("chunk: ($px, $py)-($width, $height)")
                progress(currentPixels, totalPixels)
                if (width >= pad && height >= pad) {
                    val chunk = bmpPad.copySliceWithSize(px, py, width, height)
                    //val chunk2 = chunk.resize(chunkSize, chunkSize)
                    val out = waifu2x(chunk, *channels)
                    val px = cx * (chunkSize - pad)
                    val py = cy * (chunkSize - pad)
                    //println("$currentPixels/$totalPixels")

                    //println("--------")
                    //println(bmpOut)
                    //println(chunk)
                    //println(chunk2)
                    //println(out)
                    //println("$px, $py")
                    //println("$width, $height")
                    bmpOut._draw(
                        out,
                        px,
                        py,
                        0,
                        0,
                        min(min(out.width, width), bmpOut.width - px),
                        min(min(out.height, height), bmpOut.height - py),
                        false
                    )
                }
                currentPixels += (width - pad) * (height - pad)
            }
        }
        progress(totalPixels, totalPixels)

        return bmpOut.copySliceWithSize(0, 0, bmpYCbCr.width, bmpYCbCr.height)
    }

    fun waifu2xChunkedRgba(
        bmp: Bitmap32,
        vararg channels: BitmapChannel = arrayOf(
            BitmapChannel.RED,
            BitmapChannel.GREEN,
            BitmapChannel.BLUE,
            BitmapChannel.ALPHA
        ),
        progress: (current: Int, total: Int) -> Unit = { current, total -> }
    ): Bitmap32 {
        return waifu2xChunkedYCbCr(
            bmp.rgbaToYCbCr(),
            *channels,
            progress = progress
        ).yCbCrToRgba()
    }

    suspend fun waifu2xChunkedRgbaFast(
        bmp: Bitmap32,
        vararg channels: BitmapChannel = arrayOf(
            BitmapChannel.RED,
            BitmapChannel.GREEN,
            BitmapChannel.BLUE,
            BitmapChannel.ALPHA
        ),
        progress: (current: Int, total: Int) -> Unit = { current, total -> }
    ) = waifu2xChunkedRgba(bmp.scaleNearest(2, 2), *channels, progress = progress)

    override fun close() {
        ctx.close()
    }
}

fun Bitmap32.scaleNearestFloat(sx: Float, sy: Float): Bitmap32 {
    val out = Bitmap32((width * sx).toInt(), (height * sy).toInt())
    val isx = 1f / sx
    val isy = 1f / sy
    for (y in 0 until out.height) {
        for (x in 0 until out.width) {
            out[x, y] = this[(x * isx).toInt(), (y * isy).toInt()]
        }
    }
    return out
}

private fun Bitmap32.resize(ewidth: Int, eheight: Int): Bitmap32 {
    val out = Bitmap32(ewidth, eheight)
    out.put(this)
    return out
}

// @TODO: Move to korio and korim
private operator fun Bitmap32.get(xrange: IntRange, yrange: IntRange): Bitmap32 {
    return this.copySliceWithSizeOutOfBounds(xrange.start, yrange.start, xrange.length, yrange.length)
}

private val IntRange.length: Int get() = endInclusive - start + 1

private fun Bitmap32.pad(left: Int, top: Int = left, right: Int = left, bottom: Int = top): Bitmap32 {
    val out = Bitmap32(width + left + right, height + top + bottom)
    out.put(this, left, top)
    return out
}

operator fun FloatArray2.get(xrange: IntRange, yrange: IntRange): FloatArray2 {
    val out = FloatArray2(xrange.length, yrange.length)
    for (y in 0 until out.height) {
        for (x in 0 until out.width) {
            out[x, y] = this[x + xrange.start, y + yrange.start]
        }
    }
    return out
}

fun FloatArray2.toBmp32(): Bitmap32 {
    val out = Bitmap32(width, height)
    for (n in 0 until out.area) {
        val v = (this.data[n] * 255f).clamp(0f, 255f).toInt()
        out.data[n] = RGBA.pack(v, v, v, 0xFF)
    }
    return out
}

// Exposed functions
object Kaifu2x {
    suspend fun noiseReductionRgba(
        image: Bitmap32,
        noise: Int,
        channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A),
        parallel: Boolean = true,
        chunkSize: Int = 128,
        output: PrintStream? = System.err
    ): Bitmap32 {
        //return getNoiseModel(noise, output)?.waifu2xCoreRgba("noise$noise", image, channels, parallel, chunkSize, output) ?: image
        val noiseModel = getNoiseModel(noise, output) ?: return image
        return processMeasurer("noise$noise", output) { progress ->
            Kaifu2xOpencl(noiseModel, chunkSize).use {
                it.waifu2xChunkedRgba(
                    image,
                    *channels.toTypedArray(),
                    progress = progress
                )
            }
        }.apply {
            output?.println()
        }
    }

    val scale2x by lazy {
        Kaifu2xOpencl(getScale2xModel(System.err), chunkSize = 128)
    }

    suspend fun scaleRgba(
        image: Bitmap32,
        scale: Int,
        channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A),
        output: PrintStream? = System.err
    ): Bitmap32 {
        return when (scale) {
            1 -> image
            2 -> {
                processMeasurer("scale$scale", output) { progress ->
                    scale2x.waifu2xChunkedRgbaFast(image, *channels.toTypedArray(), progress = progress)
                }.apply {
                    output?.println()
                }
            }
            else -> invalidArg("Invalid scale $scale")
        }
    }
}

suspend fun <T> processMeasurer(
    name: String,
    output: PrintStream? = System.out,
    callback: suspend (progress: (Int, Int) -> Unit) -> T
): T {
    val startTime = System.currentTimeMillis()

    return callback { current, total ->
        if (output != null) {
            val currentTime = System.currentTimeMillis()
            val ratio = current.toDouble() / total.toDouble()
            val elapsedMs = (currentTime - startTime)
            val estimatedMs = elapsedMs * (1.0 / ratio)
            output.print(
                "\r[%s] %.1f%% - ELA: %s - ETA: %s - MEM: %s ".format(
                    name,
                    (ratio * 100).toFloat(),
                    TimeSpan.toTimeString(elapsedMs.toInt()),
                    TimeSpan.toTimeString((estimatedMs - elapsedMs).toInt()),
                    getMemoryUsedString()
                )
            )
        }
    }
}

fun getMemoryUsedString(): String = "%.2f MB".format(getMemoryUsed().toDouble() / (1024.0 * 1024.0))
fun getMemoryUsed(): Long = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()

private val modelCache = LinkedHashMap<String, Model>()
internal fun readModel(name: String, output: PrintStream? = System.err): Model = modelCache.getOrPut(name) {
    output?.print("Reading $name...")
    val jsonString = Kaifu2x::class.java.getResourceAsStream("/models/$name").readBytes().toString(ASCII)
    //val jsonString = ClassLoader.getSystemClassLoader().getResourceAsStream("models/$name").readBytes().toString(ASCII)
    val json = Json.decode(jsonString)
    return Model.parseJson(json).apply {
        output?.println("Ok")
    }
}

internal fun getScale2xModel(output: PrintStream? = System.err): Model = readModel("scale2.0x_model.json", output)
internal fun getNoiseModel(level: Int, output: PrintStream? = System.err): Model? =
    if (level in 1..3) readModel("noise${level}_model.json", output) else null

object HalfFloat {
    // ignores the higher 16 bits
    fun toFloat(hbits: Int): Float {
        var mant = hbits and 0x03ff            // 10 bits mantissa
        var exp = hbits and 0x7c00            // 5 bits exponent
        if (exp == 0x7c00)
        // NaN/Inf
            exp = 0x3fc00                    // -> NaN/Inf
        else if (exp != 0)
        // normalized value
        {
            exp += 0x1c000                   // exp - 15 + 127
            if (mant == 0 && exp > 0x1c400)
            // smooth transition
                return java.lang.Float.intBitsToFloat(
                    hbits and 0x8000 shl 16
                            or (exp shl 13) or 0x3ff
                )
        } else if (mant != 0)
        // && exp==0 -> subnormal
        {
            exp = 0x1c400
            do {
                mant = mant shl 1
                exp -= 0x400
            } while ((mant and 0x400) == 0)
            mant = mant and 0x3ff
        }
        return java.lang.Float.intBitsToFloat(
            (((hbits and 0x8000) shl 16) or ((exp or mant) shl 13))
        )
    }

    // returns all higher 16 bits as 0 for all results
    fun fromFloat(fval: Float): Int {
        val fbits = java.lang.Float.floatToIntBits(fval)
        val sign = fbits.ushr(16) and 0x8000          // sign only
        var `val` = (fbits and 0x7fffffff) + 0x1000 // rounded value

        if (`val` >= 0x47800000)
        // might be or become NaN/Inf
        {                                     // avoid Inf due to rounding
            return if (fbits and 0x7fffffff >= 0x47800000) {                                 // is or must become NaN/Inf
                if (`val` < 0x7f800000) sign or 0x7c00 else sign or 0x7c00 or        // remains +/-Inf or NaN

                        (fbits and 0x007fffff).ushr(13)     // make it +/-Inf
                // keep NaN (and Inf) bits
            } else sign or 0x7bff
// unrounded not quite Inf
        }
        if (`val` >= 0x38800000)
        // remains normalized value
            return sign or (`val` - 0x38000000).ushr(13) // exp - 127 + 15
        if (`val` < 0x33000000)
        // too small for subnormal
            return sign                      // becomes +/-0
        `val` = (fbits and 0x7fffffff).ushr(23)  // tmp exp for subnormal calc
        return sign or ((fbits and 0x7fffff or 0x800000) // add subnormal bit
                + 0x800000.ushr(`val` - 102))     // round depending on cut off
            .ushr(126 - `val`)   // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
    }
}