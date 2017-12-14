package com.soywiz.kaifu2x

import com.soywiz.kaifu2x.util.*
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.format.defaultImageFormats
import com.soywiz.korim.format.readBitmapNoNative
import com.soywiz.korim.format.registerStandard
import com.soywiz.korim.format.writeTo
import com.soywiz.korio.Korio
import com.soywiz.korio.error.invalidArg
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.vfs.LocalVfs
import com.soywiz.korio.vfs.PathInfo
import com.soywiz.korio.vfs.resourcesVfs
import java.io.File
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.system.measureTimeMillis

fun main(args: Array<String>) = Kaifu2xCli.main(args)

object Kaifu2xCli {
	fun help() {
		System.err.println("Usage: kaifu2x [switches] <input.png> <output.png>")
		System.err.println("")
		System.err.println("Available switches:")
		System.err.println("  -h      - Displays this help")
		System.err.println("  -n[0-3] - Noise reduction [default to 0 (no noise reduction)]")
		System.err.println("  -s[1-2] - Scale level 1=1x, 2=2x [default to 1 (no scale)]")
		System.err.println("  -mt     - Multi Threaded [default]")
		System.err.println("  -st     - Single Threaded")
		System.err.println("  -cl     - Process Luminance")
		System.err.println("  -cla    - Process Luminance & Alpha [default]")
		System.err.println("  -clca   - Process Luminance & Chroma & Alpha")
	}

	fun helpAndExit(code: Int = -1) = run { help(); System.exit(code) }

	@JvmStatic
	fun main(args: Array<String>) = Korio {
		if (args.size < 2) helpAndExit()

		var parallel = true
		var components = listOf(ColorComponent.RED, ColorComponent.ALPHA)
		var inputName: String? = null
		var outputName: String? = null
		var noiseReduction: Int = 0
		var scale: Int = 1

		val argsR = LinkedList(args.toList())
		while (argsR.isNotEmpty()) {
			val c = argsR.removeFirst()
			when (c) {
				"-h" -> helpAndExit()
				"-st" -> parallel = false
				"-mt" -> parallel = true
				"-n0" -> noiseReduction = 0
				"-n1" -> noiseReduction = 1
				"-n2" -> noiseReduction = 2
				"-n3" -> noiseReduction = 3
				"-s1" -> scale = 1
				"-s2" -> scale = 2
				"-cl" -> components = listOf(ColorComponent.RED)
				"-cla" -> components = listOf(ColorComponent.RED, ColorComponent.ALPHA)
				"-clca" -> components = ColorComponent.ALL.toList()
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

		if (outputExtension != "png") invalidOp("Just supported png outputs but found extension $outputExtension")

		defaultImageFormats.registerStandard()
		System.err.print("Reading $inputFileName...")
		val image = LocalVfs(File(inputFileName)).readBitmapNoNative().toBMP32()
		System.err.println("Ok")

		val noiseReductedImage = Kaifu2x.noiseReductionRgba(image, noiseReduction, components, parallel)
		val scaledImage = Kaifu2x.scaleRgba(noiseReductedImage, scale, components, parallel)

		val outFile = LocalVfs(File(outputFileName)).ensureParents()
		System.err.print("Writting $outputFileName...")
		scaledImage.writeTo(outFile)
		System.err.println("Ok")

		if (noiseReduction == 0 && scale == 1) {
			System.err.println("WARNING!!: No operation done! Please add -nX or -sX switches to control noise reduction and scaling")
		}
	}
}

// Exposed functions
object Kaifu2x {
	suspend fun noiseReductionRgba(image: Bitmap32, noise: Int, components: List<ColorComponent>, parallel: Boolean): Bitmap32 {
		return getNoiseModel(noise)?.waifu2xCoreRgba("noise$noise", image, components, parallel) ?: image
	}

	suspend fun scaleRgba(image: Bitmap32, scale: Int, components: List<ColorComponent>, parallel: Boolean): Bitmap32 {
		return when (scale) {
			1 -> image
			2 -> getScale2xModel().waifu2xCoreRgba("scale$scale", image.scaleNearest(scale, scale), components, parallel)
			else -> invalidArg("Invalid scale $scale")
		}
	}
}

suspend fun Model.waifu2xCoreRgba(name: String, image: Bitmap32, components: List<ColorComponent>, parallel: Boolean): Bitmap32 {
	val model = this
	val imYCbCr = image.rgbaToYCbCr()
	val time = measureTimeMillis {
		System.err.print("Computing relevant components...\r")
		val acomponents = components.filter { imYCbCr.readComponentf(it).run { !areAllEqualTo(this[0, 0]) } }

		System.err.println("Components: Requested:${components.map { it.toStringYCbCr() }} -> Required:${acomponents.map { it.toStringYCbCr() }}")

		val startTime = System.currentTimeMillis()
		model.waifu2xYCbCrInplace(imYCbCr, acomponents, parallel = parallel) { current, total ->
			val currentTime = System.currentTimeMillis()
			val ratio = current.toDouble() / total.toDouble()
			val elapsedMs = (currentTime - startTime)
			val estimatedMs = elapsedMs * (1.0 / ratio)
			System.err.print(
				"\rProgress($name): %.1f%% - Elapsed: %s - Remaining: %s  ".format(
					(ratio * 100).toFloat(),
					toTimeString(elapsedMs.toInt()),
					toTimeString((estimatedMs - elapsedMs).toInt())
				)
			)
		}
	}
	System.err.println()
	System.err.println("Took: " + time.toDouble() / 1000 + " seconds")
	return imYCbCr.yCbCrToRgba()
}

//fun Model.waifu2xRgba(imRgba: Bitmap32, acomponents: List<ColorComponent>, parallel: Boolean = true, progressReport: (Int, Int) -> Unit = { cur, total -> }): Bitmap32 {
//	val imYCbCr = imRgba.rgbaToYCbCr()
//	val result = waifu2xYCbCrInplace(imYCbCr, acomponents, parallel, progressReport)
//	return result.yCbCrToRgba()
//}

fun Model.waifu2xYCbCrInplace(imYCbCr: Bitmap32, acomponents: List<ColorComponent>, parallel: Boolean = true, progressReport: (Int, Int) -> Unit = { cur, total -> }): Bitmap32 {
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
