package com.soywiz.kaifu2x

import com.soywiz.kaifu2x.util.FloatArray2
import com.soywiz.kaifu2x.util.readComponentf
import com.soywiz.kaifu2x.util.writeComponentf
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
import com.soywiz.korio.vfs.LocalVfs
import com.soywiz.korio.vfs.PathInfo
import com.soywiz.korio.vfs.resourcesVfs
import java.io.File
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
		var components = listOf(BitmapChannel.Y, BitmapChannel.A)
		var inputName: String? = null
		var outputName: String? = null
		var noiseReduction: Int = 0
		var scale: Int = 1
		var quality: Int = 100

		if (args.isEmpty()) helpAndExit()

		val argsR = LinkedList(args.toList())
		while (argsR.isNotEmpty()) {
			val c = argsR.removeFirst()
			when {
				c == "-h" -> helpAndExit()
				c == "-v" -> run { println(KAIFU2X_VERSION); exitProcess(-1) }
				c == "-st" -> parallel = false
				c == "-mt" -> parallel = true
				c == "-n0" -> noiseReduction = 0
				c == "-n1" -> noiseReduction = 1
				c == "-n2" -> noiseReduction = 2
				c == "-n3" -> noiseReduction = 3
				c == "-s1" -> scale = 1
				c == "-s2" -> scale = 2
				c == "-cl" -> components = listOf(BitmapChannel.Y)
				c == "-cla" -> components = listOf(BitmapChannel.Y, BitmapChannel.A)
				c == "-clca" -> components = BitmapChannel.ALL.toList()
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

		val inputFileName = inputName ?: invalidOp("Missing input file name")
		val outputFileName = outputName ?: invalidOp("Missing output file name")

		val outputExtension = PathInfo(outputFileName).extensionLC

		if (outputExtension !in listOf("png", "jpg")) invalidOp("Just supported 'png' or 'jpg' outputs but found extension $outputExtension")

		defaultImageFormats.registerStandard()
		System.err.print("Reading $inputFileName...")
		val image = LocalVfs(File(inputFileName)).readBitmapNoNative().toBMP32()
		System.err.println("Ok")

		val noiseReductedImage = Kaifu2x.noiseReductionRgba(image, noiseReduction, components, parallel)
		val scaledImage = Kaifu2x.scaleRgba(noiseReductedImage, scale, components, parallel)

		val outFile = LocalVfs(File(outputFileName)).ensureParents()
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
	suspend fun noiseReductionRgba(image: Bitmap32, noise: Int, components: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true): Bitmap32 {
		return getNoiseModel(noise)?.waifu2xCoreRgba("noise$noise", image, components, parallel) ?: image
	}

	suspend fun scaleRgba(image: Bitmap32, scale: Int, components: List<BitmapChannel> = listOf(BitmapChannel.Y, BitmapChannel.A), parallel: Boolean = true): Bitmap32 {
		return when (scale) {
			1 -> image
			2 -> getScale2xModel().waifu2xCoreRgba("scale$scale", image.scaleNearest(scale, scale), components, parallel)
			else -> invalidArg("Invalid scale $scale")
		}
	}
}

suspend fun Model.waifu2xCoreRgba(name: String, image: Bitmap32, components: List<BitmapChannel>, parallel: Boolean): Bitmap32 {
	//val chunkSize = 128
	//val chunkSize = 32
	//val chunkSize = 400
	//val artifactThresold = 2
	//val artifactThresold = 4
	//val artifactThresold = 8
	//val artifactThresold = 16
	//val artifactThresold = 32

	val chunkSize = 2048
	//val chunkSize = 128
	val artifactThresold = 0
	//val artifactThresold = 32
	//val artifactThresold = 40

	val model = this
	val imYCbCr = image.rgbaToYCbCr()
	val time = measureTimeMillis {
		System.err.print("Computing relevant components...\r")
		val acomponents = components.filter { imYCbCr.readComponentf(it).run { !areAllEqualTo(this[0, 0]) } }

		System.err.println("Components: Requested:${components.map { it.toStringYCbCr() }} -> Required:${acomponents.map { it.toStringYCbCr() }}")

		for (y in 0 until imYCbCr.height step (chunkSize - artifactThresold * 2)) {
			for (x in 0 until imYCbCr.width step (chunkSize - artifactThresold * 2)) {
				val swidth = min(chunkSize, imYCbCr.width - x)
				val sheight = min(chunkSize, imYCbCr.height - y)
				println("CHUNK($x, $y, $swidth, $sheight) [${imYCbCr.width}, ${imYCbCr.height}]")
				val chunk = imYCbCr.copySliceWithSize(x, y, swidth, sheight)

				System.gc()

				val startTime = System.currentTimeMillis()
				model.waifu2xYCbCrInplace(chunk, acomponents, parallel = parallel) { current, total ->
					val currentTime = System.currentTimeMillis()
					val ratio = current.toDouble() / total.toDouble()
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
				System.err.println()

				val dx = if (x == 0) 0 else artifactThresold
				val dy = if (y == 0) 0 else artifactThresold
				if ((chunk.width - dx) > 0 && (chunk.height - dy) > 0) {
					imYCbCr.put(chunk.copySliceWithSize(dx, dy, chunk.width - dx, chunk.height - dy), x + dx, y + dy)
					//imYCbCr.put(chunk.sliceWithBounds(dx, dy, chunk.width, chunk.height), x + dx, y + dy)
					//imYCbCr.put(chunk.copySliceWithSize2(artifactThresold, artifactThresold, chunk.width - artifactThresold, chunk.height - artifactThresold), x + artifactThresold, y + artifactThresold)
				}
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

//fun Model.waifu2xRgba(imRgba: Bitmap32, acomponents: List<ColorComponent>, parallel: Boolean = true, progressReport: (Int, Int) -> Unit = { cur, total -> }): Bitmap32 {
//	val imYCbCr = imRgba.rgbaToYCbCr()
//	val result = waifu2xYCbCrInplace(imYCbCr, acomponents, parallel, progressReport)
//	return result.yCbCrToRgba()
//}

fun Model.waifu2xYCbCrInplace(imYCbCr: Bitmap32, acomponents: List<BitmapChannel>, parallel: Boolean = true, progressReport: (Int, Int) -> Unit = { cur, total -> }): Bitmap32 {
	val nthreads = if (parallel) Runtime.getRuntime().availableProcessors() else 1
	System.err.println("Processing Threads: $nthreads")

	for ((index, c) in acomponents.withIndex()) {
		val data = imYCbCr.readComponentf(c)
		val result = waifu2xCore(data, nthreads = nthreads) { current, total ->
			val rcurrent = index * total + current
			val rtotal = total * acomponents.size
			progressReport(rcurrent, rtotal)
		}
		imYCbCr.writeComponentf(c, result)
	}
	return imYCbCr
}

fun Model.waifu2xCore(map: FloatArray2, nthreads: Int, progressReport: (Int, Int) -> Unit = { cur, total -> }): FloatArray2 {
	var i_planes = arrayOf(map.paddedEdge(steps.size))
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


private val modelsVfs by lazy { resourcesVfs["models"] }

private suspend fun readModel(name: String): Model {
	System.err.print("Reading $name...")
	val jsonString = modelsVfs[name].readBytes().toString(ASCII)
	val json = Json.decode(jsonString)
	return Model.parseJson(json).apply {
		System.err.println("Ok")
	}
}

private suspend fun getScale2xModel(): Model = readModel("scale2.0x_model.json")
private suspend fun getNoiseModel(level: Int): Model? = if (level in 1..3) readModel("noise${level}_model.json") else null
