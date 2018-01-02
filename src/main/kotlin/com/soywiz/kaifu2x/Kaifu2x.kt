package com.soywiz.kaifu2x

import com.soywiz.kaifu2x.util.*
import com.soywiz.klock.TimeSpan
import com.soywiz.korim.bitmap.A
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korim.bitmap.Y
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.format.*
import com.soywiz.korio.Korio
import com.soywiz.korio.error.invalidArg
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.util.clamp
import com.soywiz.korio.util.substr
import com.soywiz.korio.util.toIntCeil
import com.soywiz.korio.vfs.PathInfo
import com.soywiz.korio.vfs.UniversalVfs
import com.soywiz.korio.vfs.localCurrentDirVfs
import java.io.Closeable
import java.io.PrintStream
import java.util.*
import kotlin.math.min
import kotlin.system.exitProcess

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
        System.err.println("  -mt       - Multi Threaded [default]")
        System.err.println("  -st       - Single Threaded")
        System.err.println("  -cl       - Process Luminance")
        System.err.println("  -cla      - Process Luminance & Alpha")
        System.err.println("  -clca     - Process Luminance & Chroma & Alpha [default]")
    }

    fun helpAndExit(code: Int = -1) = run { help(); System.exit(code) }

    @JvmStatic
    fun main(args: Array<String>) = Korio {
        var parallel = true
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
                c == "-st" -> parallel = false
                c == "-mt" -> parallel = true
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

        if (outputExtension !in listOf("png", "jpg")) invalidOp("Just supported 'png' or 'jpg' outputs but found extension $outputExtension")

        defaultImageFormats.registerStandard()
        System.err.print("Reading $inputFileName...")
        val image = UniversalVfs(inputFileName).readBitmapNoNative().toBMP32()
        System.err.println("Ok")

        val noiseReductedImage = Kaifu2x.noiseReductionRgba(image, noiseReduction, channels, parallel, chunkSize = chunkSize)
        val scaledImage = Kaifu2x.scaleRgba(noiseReductedImage, scale, channels, parallel, chunkSize = chunkSize)

        val outFile = UniversalVfs(outputFileName).ensureParents()
        System.err.print("Writting $outputFileName...")
        scaledImage.writeTo(outFile, ImageEncodingProps(quality = quality.toDouble() / 100.0))
        System.err.println("Ok")

        if (noiseReduction == 0 && scale == 1) {
            System.err.println("WARNING!!: No operation done! Please add -nX or -sX switches to control noise reduction and scaling")
        }
    }
}

class Kaifu2xOpencl(private val model: Model) : Closeable {
    private val steps = model.steps
    private val ctx = ClContext()
    private val queue = ctx.createCommandQueue()
    private val program = ctx.createProgram("""
        __kernel void waifu2x(
            const unsigned int width,
            const unsigned int height,
            const unsigned int num_inputs,
            const unsigned int num_outputs,
            global const float *in,
            global const float *krn,
            global const float *bias,
            global float *out
        ) {
            int area = width * height;
            int x = get_global_id(0);
            int y = get_global_id(1);
            int z = get_global_id(2);
            int xy = x + (y * width);

            float acc = bias[z];
            int krnOffset = z * num_inputs * 9;
            for (int i = 0; i < num_inputs; i++) {
                int i_off = xy + (i * area);
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        acc += in[i_off + (width * j) + k] * krn[krnOffset + (i * 9) + (j * 3) + k];
                    }
                }
            }
            out[xy + (z * area)] = acc - 0.9 * fmin(acc, 0);
        }
    """)
    private val waifu2x = program["waifu2x"]
    private val biasPerStep = model.steps.map { ctx.createBuffer(it.bias) }
    private val kernelPerSteps = model.steps.map { ctx.createBuffer(it.weight.flatMap { it.flatMap { it.toList() } }.toFloatArray()) }

    // data already have padding
    fun waifu2x(data: FloatArray2): FloatArray2 {
        val bw = data.width
        val bh = data.height
        val pad = model.padding
        val out = queue {
            finish()

            var ninput = ctx.createBuffer(data.data)
            //val buffers = listOf(ctx.createBuffer(data.data)) + steps.map { ctx.createEmptyBuffer(4, bw * bh * it.nOutputPlane) }
            lateinit var noutput: ClBuffer


            for (i in 0 until steps.size) {
                val n_in = steps[i].nInputPlane
                val n_out = steps[i].nOutputPlane
                val bias_buf = biasPerStep[i]
                val kern_buf = kernelPerSteps[i]

                //val ninput = buffers[i]
                //noutput = buffers[i + 1]
                noutput = ctx.createEmptyBuffer(4, bw * bh * n_out)

                //println("i=$i, n_out=$n_out, n_in=$n_in, bw=$bw, bh=$bh, bias_buf=${bias_buf.length}, kern_buf=${kern_buf.length}")

                waifu2x(
                        queue,
                        bw, bh, n_in, n_out,
                        ninput, kern_buf, bias_buf, noutput,
                        globalWorkRanges = listOf(
                                0L until (bw - (2 * i) - 2),
                                0L until (bh - (2 * i) - 2),
                                0L until n_out.toLong()
                        )
                )

                ninput.close()
                ninput = noutput
            }

            noutput.readFloats().apply {
                noutput.close()
            }
        }

        //println(out.toTypedArray().size)
        //return FloatArray2(bw - pad * 2, bh - pad * 2, out.toTypedArray())
        return FloatArray2(bw, bh, out.toTypedArray())[0 until (bw - pad * 2), 0 until (bh - pad * 2)]
    }

    fun waifu2x(data: Bitmap32, vararg channels: BitmapChannel = arrayOf(BitmapChannel.RED, BitmapChannel.GREEN, BitmapChannel.BLUE, BitmapChannel.ALPHA)): Bitmap32 {
        val out = data[7 until data.width - 7, 7 until data.height - 7]
        for (channel in channels) {
            val i = data.readChannelf(channel)
            val o = waifu2x(i)
            out.writeChannelf(channel, o)
        }
        return out
    }

    fun waifu2xChunkedYCbCr(bmpYCbCr: Bitmap32, vararg channels: BitmapChannel = arrayOf(BitmapChannel.RED, BitmapChannel.GREEN, BitmapChannel.BLUE, BitmapChannel.ALPHA), chunkSize: Int = 128, progress: (current: Int, total: Int) -> Unit = { current, total -> }): Bitmap32 {
        val pad = steps.size * 2
        val bmpPad = bmpYCbCr.pad(pad / 2)
        val bmpOut = Bitmap32(bmpPad.width - pad, bmpPad.height - pad)
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
                    val out = waifu2x(chunk, *channels)
                    //println("$currentPixels/$totalPixels")
                    bmpOut.put(out, cx * (chunkSize - pad), cy * (chunkSize - pad))
                }
                currentPixels += (width - pad) * (height - pad)
            }
        }
        progress(currentPixels, totalPixels)

        return bmpOut
    }

    fun waifu2xChunkedRgba(bmp: Bitmap32, vararg channels: BitmapChannel = arrayOf(BitmapChannel.RED, BitmapChannel.GREEN, BitmapChannel.BLUE, BitmapChannel.ALPHA), chunkSize: Int = 128, progress: (current: Int, total: Int) -> Unit = { current, total -> }): Bitmap32 {
        return waifu2xChunkedYCbCr(bmp.rgbaToYCbCr(), *channels, chunkSize = chunkSize, progress = progress).yCbCrToRgba()
    }

    override fun close() {
        queue.close()
        ctx.close()
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) = Korio {
            val bmp = localCurrentDirVfs["src/test/resources/goku_small_bg.png"].readBitmapNoNative().toBMP32()
            val kaifu2x = Kaifu2xOpencl(getScale2xModel())

            val bmpOut = kaifu2x.waifu2xChunkedRgba(bmp.rgbaToYCbCr().scaleNearest(2, 2)) { current, total ->
                println("$current/$total")
            }

            localCurrentDirVfs["pad.png"].writeBitmap(bmpOut.yCbCrToRgba())
        }
    }
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
    suspend fun noiseReductionRgba(image: Bitmap32, noise: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128, output: PrintStream? = System.err): Bitmap32 {
        //return getNoiseModel(noise, output)?.waifu2xCoreRgba("noise$noise", image, channels, parallel, chunkSize, output) ?: image
        val noiseModel = getNoiseModel(noise, output) ?: return image
        return processMeasurer("noise$noise", output) { progress ->
            Kaifu2xOpencl(noiseModel).use { it.waifu2xChunkedRgba(image, *channels.toTypedArray(), chunkSize = chunkSize, progress = progress) }
        }
    }

    suspend fun scaleRgba(image: Bitmap32, scale: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128, output: PrintStream? = System.err): Bitmap32 {
        return when (scale) {
            1 -> image
            2 -> {
                processMeasurer("scale$scale", output) { progress ->
                    Kaifu2xOpencl(getScale2xModel(output)!!).use { it.waifu2xChunkedRgba(image.scaleNearest(2, 2), *channels.toTypedArray(), chunkSize = chunkSize, progress = progress) }
                }
            }
            else -> invalidArg("Invalid scale $scale")
        }
    }
}

fun <T> processMeasurer(name: String, output: PrintStream? = System.out, callback: (progress: (Int, Int) -> Unit) -> T): T {
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

fun getMemoryUsedString(): String {
    return "%.2f MB".format(getMemoryUsed().toDouble() / (1024.0 * 1024.0))
}

fun getMemoryUsed(): Long {
    return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
}

internal fun readModel(name: String, output: PrintStream? = System.err): Model {
    output?.print("Reading $name...")
    val jsonString = Kaifu2x::class.java.getResourceAsStream("/models/$name").readBytes().toString(ASCII)
    //val jsonString = ClassLoader.getSystemClassLoader().getResourceAsStream("models/$name").readBytes().toString(ASCII)
    val json = Json.decode(jsonString)
    return Model.parseJson(json).apply {
        output?.println("Ok")
    }
}

internal fun getScale2xModel(output: PrintStream? = System.err): Model = readModel("scale2.0x_model.json", output)
internal fun getNoiseModel(level: Int, output: PrintStream? = System.err): Model? = if (level in 1..3) readModel("noise${level}_model.json", output) else null
