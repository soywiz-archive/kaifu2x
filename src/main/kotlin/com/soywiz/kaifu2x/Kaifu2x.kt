package com.soywiz.kaifu2x

import com.soywiz.kaifu2x.util.FloatArray2
import com.soywiz.kaifu2x.util.copySliceWithSizeOutOfBounds
import com.soywiz.kaifu2x.util.readChannelf
import com.soywiz.kaifu2x.util.writeChannelf
import com.soywiz.klock.TimeSpan
import com.soywiz.korim.bitmap.*
import com.soywiz.korim.format.*
import com.soywiz.korio.Korio
import com.soywiz.korio.error.invalidArg
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.util.substr
import com.soywiz.korio.vfs.*
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
		System.err.println("  -cla      - Process Luminance & Alpha [default]")
		System.err.println("  -clca     - Process Luminance & Chroma & Alpha")
	}

	fun helpAndExit(code: Int = -1) = run { help(); System.exit(code) }

	@JvmStatic
	fun main(args: Array<String>) = Korio {
		var parallel = true
		var channels = listOf(BitmapChannel.Y, BitmapChannel.A)
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

		val noiseReductedImage = Kaifu2x.noiseReductionRgba(image, noiseReduction, channels, parallel)
		val scaledImage = Kaifu2x.scaleRgba(noiseReductedImage, scale, channels, parallel)

		val outFile = UniversalVfs(outputFileName).ensureParents()
		System.err.print("Writting $outputFileName...")
		scaledImage.writeTo(outFile, ImageEncodingProps(quality = quality.toDouble() / 100.0))
		System.err.println("Ok")

		if (noiseReduction == 0 && scale == 1) {
			System.err.println("WARNING!!: No operation done! Please add -nX or -sX switches to control noise reduction and scaling")
		}
	}
}

// Exposed functions
object Kaifu2x {
	suspend fun noiseReductionRgba(image: Bitmap32, noise: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128): Bitmap32 {
		return getNoiseModel(noise)?.waifu2xCoreRgba("noise$noise", image, channels, parallel, chunkSize) ?: image
	}

	suspend fun scaleRgba(image: Bitmap32, scale: Int, channels: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true, chunkSize: Int = 128): Bitmap32 {
		return when (scale) {
			1 -> image
			2 -> getScale2xModel().waifu2xCoreRgba("scale$scale", image.scaleNearest(scale, scale), channels, parallel, chunkSize)
			else -> invalidArg("Invalid scale $scale")
		}
	}
}

suspend fun Model.waifu2xCoreRgba(name: String, image: Bitmap32, channels: List<BitmapChannel>, parallel: Boolean, chunkSize: Int): Bitmap32 {
	val model = this
	val imYCbCr = image.rgbaToYCbCr()
	val padding = this.padding

	val time = measureTimeMillis {
		System.err.print("Computing relevant channels...\r")
		val achannels = channels.filter { imYCbCr.readChannelf(it).run { !areAllEqualTo(this[0, 0]) } }

		System.err.println("Channels: Requested${channels.map { it.toStringYCbCr() }} -> Required${achannels.map { it.toStringYCbCr() }}")
		val nthreads = if (parallel) Runtime.getRuntime().availableProcessors() else 1
		System.err.println("Chunk size: $chunkSize, Threads: $nthreads")

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
					val currentTime = System.currentTimeMillis()
					val localRatio = current.toDouble() / total.toDouble()
					val localProcessedPixels = (chunkPixels * localRatio).toInt()
					val totalProcessedPixels = processedPixels + localProcessedPixels
					val ratio = totalProcessedPixels.toDouble() / totalPixels.toDouble()
					val elapsedMs = (currentTime - startTime)
					val estimatedMs = elapsedMs * (1.0 / ratio)
					System.err.print(
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
				//System.err.println()

				//println("outUnpaddedChunk: $outUnpaddedChunk -> $x, $y")
				imYCbCr.put(outUnpaddedChunk, x, y)
			}
		}
	}
	System.err.println()
	System.err.println("Took: " + time.toDouble() / 1000 + " seconds")
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
		val result = waifu2xCore(data, nthreads = nthreads, addPadding = false) { current, total ->
			val rcurrent = index * total + current
			val rtotal = total * achannels.size
			progressReport(rcurrent, rtotal)
		}
		out.writeChannelf(c, result)
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


private fun readModel(name: String): Model {
	System.err.print("Reading $name...")
	val jsonString = Kaifu2x::class.java.getResourceAsStream("/models/$name").readBytes().toString(ASCII)
	//val jsonString = ClassLoader.getSystemClassLoader().getResourceAsStream("models/$name").readBytes().toString(ASCII)
	val json = Json.decode(jsonString)
	return Model.parseJson(json).apply {
		System.err.println("Ok")
	}
}

private fun getScale2xModel(): Model = readModel("scale2.0x_model.json")
private fun getNoiseModel(level: Int): Model? = if (level in 1..3) readModel("noise${level}_model.json") else null
