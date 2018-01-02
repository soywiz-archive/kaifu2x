package com.soywiz.kaifu2x

import com.soywiz.kaifu2x.util.*
import com.soywiz.klock.TimeSpan
import com.soywiz.korim.bitmap.*
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
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.system.exitProcess
import kotlin.system.measureTimeMillis

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

    fun waifu2xChunked(bmp: Bitmap32, vararg channels: BitmapChannel = arrayOf(BitmapChannel.RED, BitmapChannel.GREEN, BitmapChannel.BLUE, BitmapChannel.ALPHA), chunkSize: Int = 128, progress: (current: Int, total: Int) -> Unit = { current, total -> }): Bitmap32 {
        val bmpPad = bmp.pad(7)
        val bmpOut = Bitmap32(bmpPad.width - 14, bmpPad.height - 14)

        val pad = steps.size * 2
        var currentPixels = 0
        val totalPixels = bmpOut.area

        for (cy in 0 until (bmpPad.height.toDouble() / (128 - 14).toDouble()).toIntCeil()) {
            for (cx in 0 until (bmpPad.width.toDouble() / (128 - 14).toDouble()).toIntCeil()) {
                val px = cx * (128 - 14)
                val py = cy * (128 - 14)
                val width = min(128, bmpPad.width - px)
                val height = min(128, bmpPad.height - py)
                val chunk = bmpPad.copySliceWithSize(px, py, width, height)
                val out = waifu2x(chunk)
                progress(currentPixels, totalPixels)
                //println("$currentPixels/$totalPixels")
                bmpOut.put(out, cx * (128 - 14), cy * (128 - 14))
                currentPixels += (width - 14) * (height - 14)
            }
        }
        progress(currentPixels, totalPixels)

        return bmpOut
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

            val bmpOut = kaifu2x.waifu2xChunked(bmp.rgbaToYCbCr().scaleNearest(2, 2)) { current, total ->
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
        return getNoiseModel(noise, output)?.waifu2xCoreRgba("noise$noise", image, channels, parallel, chunkSize, output) ?: image
    }

    suspend fun scaleRgba(image: Bitmap32, scale: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128, output: PrintStream? = System.err): Bitmap32 {
        return when (scale) {
            1 -> image
            2 -> getScale2xModel(output).waifu2xCoreRgba("scale$scale", image.scaleNearest(scale, scale), channels, parallel, chunkSize, output)
            else -> invalidArg("Invalid scale $scale")
        }
    }
}

suspend fun Model.waifu2xCoreRgba(name: String, image: Bitmap32, channels: List<BitmapChannel>, parallel: Boolean, chunkSize: Int, output: PrintStream?): Bitmap32 {
    val model = this
    val imYCbCr = image.rgbaToYCbCr()
    val padding = this.padding

    val time = measureTimeMillis {
        output?.print("Computing relevant channels...\r")
        val achannels = channels.filter { imYCbCr.readChannelf(it).run { !areAllEqualTo(this[0, 0]) } }

        output?.println("Channels: Requested${channels.map { it.toStringYCbCr() }} -> Required${achannels.map { it.toStringYCbCr() }}")
        val nthreads = if (parallel) Runtime.getRuntime().availableProcessors() else 1
        output?.println("Chunk size: $chunkSize, Threads: $nthreads")

        var processedPixels = 0
        val totalPixels = imYCbCr.area

        val startTime = System.currentTimeMillis()
        for (y in 0 until imYCbCr.height step chunkSize) {
            for (x in 0 until imYCbCr.width step chunkSize) {
                val swidth = min(chunkSize, imYCbCr.width - x)
                val sheight = min(chunkSize, imYCbCr.height - y)
                //println("CHUNK($x, $y, $swidth, $sheight) [${imYCbCr.width}, ${imYCbCr.height}]")
                val inPaddedChunk = imYCbCr.copySliceWithSizeOutOfBounds(x - padding, y - padding, swidth + padding * 2, sheight + padding * 2)

                //println("inPaddedChunk: $inPaddedChunk")

                System.gc()

                val chunkPixels = (inPaddedChunk.width - padding * 2) * (inPaddedChunk.height - padding * 2)
                val outUnpaddedChunk = model.waifu2xYCbCrNoPadding(inPaddedChunk, achannels, nthreads = nthreads) { current, total ->
                    if (output == null) return@waifu2xYCbCrNoPadding
                    val currentTime = System.currentTimeMillis()
                    val localRatio = current.toDouble() / total.toDouble()
                    val localProcessedPixels = (chunkPixels * localRatio).toInt()
                    val totalProcessedPixels = processedPixels + localProcessedPixels
                    val ratio = totalProcessedPixels.toDouble() / totalPixels.toDouble()
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
                processedPixels += chunkPixels
                imYCbCr.put(outUnpaddedChunk, x, y)
            }
        }
    }
    output?.println()
    output?.println("Took: " + time.toDouble() / 1000 + " seconds")
    return imYCbCr.yCbCrToRgba()
}

fun getMemoryUsedString(): String {
    return "%.2f MB".format(getMemoryUsed().toDouble() / (1024.0 * 1024.0))
}

fun getMemoryUsed(): Long {
    return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
}

fun Model.waifu2xYCbCrNoPadding(imYCbCr: Bitmap32, achannels: List<BitmapChannel>, nthreads: Int, progressReport: (Int, Int) -> Unit = { cur, total -> }): Bitmap32 {
    val padding = this.padding

    val out = imYCbCr.copySliceWithBounds(padding, padding, imYCbCr.width - padding, imYCbCr.height - padding)

    for ((index, c) in achannels.withIndex()) {
        val data = imYCbCr.readChannelf(c)
        val ref = data.data[0]
        val isSolid = data.data.all { it == ref }
        if (!isSolid) {
            val result = waifu2xCore(data, nthreads = nthreads, addPadding = false) { current, total ->
                val rcurrent = index * total + current
                val rtotal = total * achannels.size
                progressReport(rcurrent, rtotal)
            }
            out.writeChannelf(c, result)
        }
    }
    return out
}

fun Model.waifu2xCore(map: FloatArray2, nthreads: Int, addPadding: Boolean = true, progressReport: (Int, Int) -> Unit = { cur, total -> }): FloatArray2 {
    var i_planes = if (addPadding) arrayOf(map.paddedEdge(steps.size)) else arrayOf(map)
    val total = steps.map { it.nInputPlane * it.nOutputPlane }.sum()
    var current = 0
    val tpool = Executors.newFixedThreadPool(nthreads)

    //progressReport(0, total)

    for (step in steps) {
        val fip = i_planes[0]
        val owidth = fip.width - 2
        val oheight = fip.height - 2
        // Do all allocations here for this step!
        val o_planes = Array(step.weight.size) { FloatArray2(owidth, oheight) }
        val tpartials = Array(nthreads) { FloatArray2(owidth, oheight) }
        val pList = Array(nthreads) { FloatArray2(owidth, oheight) }

        for (index in 0 until step.weight.size) {
            val bias = step.bias[index]
            val weights = step.weight[index]

            val futures = (0 until nthreads).map { threadId ->
                tpool.submit {
                    var first = true
                    val partial = tpartials[threadId]
                    val p = pList[threadId]
                    for (i in threadId until weights.size step nthreads) {
                        val ip = i_planes[i]
                        val kernel = weights[i]

                        p.setToConvolvedValidOptimized(ip, kernel)

                        if (first) {
                            partial.setTo(p)
                            first = false
                        } else {
                            partial.setToAdd(partial, p)
                        }
                    }
                }
            }

            val partial = o_planes[index]

            // Wait all tasks to complete
            for (n in 0 until nthreads) futures[n].get()

            // Accumulate partial values from threads
            for (n in 0 until nthreads) {
                if (n == 0) {
                    partial.setTo(tpartials[n])
                } else {
                    partial.setToAdd(partial, tpartials[n])
                }
            }

            partial.setToFunc(partial) {
                val bit = it + bias
                max(bit, 0f) + (min(bit, 0f) * 0.1f)
            }

            current += weights.size
            progressReport(current, total)
        }

        i_planes = o_planes
    }

    tpool.shutdown()

    return i_planes.first()
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
