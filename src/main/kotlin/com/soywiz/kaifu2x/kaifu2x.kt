package com.soywiz.kaifu2x

import com.soywiz.klock.TimeSpan
import com.soywiz.kmem.arraycopy
import com.soywiz.korim.bitmap.Bitmap
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.format.*
import com.soywiz.korio.Korio
import com.soywiz.korio.error.invalidArg
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.util.clamp
import com.soywiz.korio.vfs.LocalVfs
import com.soywiz.korio.vfs.PathInfo
import com.soywiz.korio.vfs.resourcesVfs
import java.io.File
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.system.measureTimeMillis

fun main(args: Array<String>) = Kaifu2x.main(args)

object Kaifu2x {
	@JvmStatic
	fun main(args: Array<String>) = Korio {
		if (args.size < 2) {
			System.err.println("Usage: kaifu2x [-st] [-mt] [-jl] [-jla] <input.png> <output.png>")
			System.exit(-1)
		}

		var multiThread = true
		var justLuminance = false
		var justLuminanceAlpha = false
		var inputName: String? = null
		var outputName: String? = null

		val argsR = LinkedList(args.toList())
		while (argsR.isNotEmpty()) {
			val c = argsR.removeFirst()
			when (c) {
				"-st" -> multiThread = false
				"-mt" -> multiThread = true
				"-jl" -> justLuminance = true
				"-jla" -> justLuminanceAlpha = true
				else -> {
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

		val model = getModel()

		val im = image.scaleNearest(2, 2)
		val imYCbCr = im.rgbaToYCbCr()
		val time = measureTimeMillis {
			val components = when {
				justLuminanceAlpha -> listOf(ColorComponent.RED, ColorComponent.ALPHA)
				justLuminance -> listOf(ColorComponent.RED)
				else -> ColorComponent.ALL.toList()
			}

			System.err.println("Input components: $components")

			val acomponents = components.filter {
				imYCbCr.readComponentf(it).run { !areAllEqualTo(this[0, 0]) }
			}

			System.err.println("Processing components: $acomponents")

			val startTime = System.currentTimeMillis()

			for ((index, c) in acomponents.withIndex()) {
				val data = imYCbCr.readComponentf(c)
				val result = model.waifu2x(data, parallel = multiThread) { current, total ->
					val currentTime = System.currentTimeMillis()
					val rcurrent = index * total + current
					val rtotal = total * acomponents.size
					val ratio = rcurrent.toDouble() / rtotal.toDouble()
					val elapsedMs = (currentTime - startTime)
					val estimatedMs = elapsedMs * (1.0 / ratio)
					System.err.print(
						"\rProgress: %.1f%% - Elapsed: %s - Remaining: %s  ".format(
							(ratio * 100).toFloat(),
							toTimeString(elapsedMs.toInt()),
							toTimeString((estimatedMs - elapsedMs).toInt())
						)
					)
				}
				imYCbCr.writeComponentf(c, result)
			}
		}
		System.err.println()
		System.err.println("Took: " + time.toDouble() / 1000 + " seconds")
		val out: Bitmap = imYCbCr.yCbCrToRgba()
		val outFile = LocalVfs(File(outputFileName)).ensureParents()
		System.err.print("Writting $outputFileName...")
		out.writeTo(outFile)
		System.err.println("Ok")
	}
}

fun TimeSpan.toTimeString(components: Int = 3): String = toTimeString(milliseconds, components)

fun toTimeString(totalMilliseconds: Int, components: Int = 3): String {
	var timeUnit = totalMilliseconds / 1000

	if (components == 1) return "%02d".format(timeUnit)

	val seconds = timeUnit % 60
	timeUnit /= 60

	if (components == 2) return "%02d:%02d".format(timeUnit, seconds)

	val minutes = timeUnit % 60
	timeUnit /= 60

	if (components == 3) return "%02d:%02d:%02d".format(timeUnit, minutes, seconds)

	TODO("Just supported 3 components")
}

object Example {
	@JvmStatic
	fun main(args: Array<String>) = Korio {
		val image = PNG.decode(resourcesVfs["samples/goku_small_bg.png"]).toBMP32()
		val image2x = image.scaleNearest(2, 2)
		val imYCbCr = image2x.rgbaToYCbCr()
		val outputVfs = LocalVfs("/tmp")

		image2x.writeTo(outputVfs["kaifu2x.nearest.2x.png"], formats = PNG)

		val YYYA = imYCbCr.clone()
		YYYA.writeComponent(ColorComponent.RED, imYCbCr, ColorComponent.RED)
		YYYA.writeComponent(ColorComponent.GREEN, imYCbCr, ColorComponent.RED)
		YYYA.writeComponent(ColorComponent.BLUE, imYCbCr, ColorComponent.RED)
		YYYA.writeTo(outputVfs["kaifu2x.YYYA.png"], formats = PNG)
		//showImageAndWait(YYYA)

		val CbCbCbA = imYCbCr.clone()
		CbCbCbA.writeComponent(ColorComponent.RED, imYCbCr, ColorComponent.GREEN)
		CbCbCbA.writeComponent(ColorComponent.GREEN, imYCbCr, ColorComponent.GREEN)
		CbCbCbA.writeComponent(ColorComponent.BLUE, imYCbCr, ColorComponent.GREEN)
		CbCbCbA.writeTo(outputVfs["kaifu2x.CbCbCbA.png"], formats = PNG)
		//showImageAndWait(CbCbCbA)

		val CrCrCrA = imYCbCr.clone()
		CrCrCrA.writeComponent(ColorComponent.RED, CrCrCrA, ColorComponent.BLUE)
		CrCrCrA.writeComponent(ColorComponent.GREEN, CrCrCrA, ColorComponent.BLUE)
		CrCrCrA.writeComponent(ColorComponent.BLUE, CrCrCrA, ColorComponent.BLUE)
		CrCrCrA.writeTo(outputVfs["kaifu2x.CrCrCrA.png"], formats = PNG)
		//showImageAndWait(CrCrCrA)

		val Y = imYCbCr.get0f()
		lateinit var result: FloatArray2
		val model = getModel()
		val time = measureTimeMillis {
			result = model.waifu2x(Y) { current, total ->
				print("\r" + ((current.toDouble() / total.toDouble()) * 100) + "%")
			}
		}

		println("Took: " + time.toDouble() / 1000 + " seconds")

		val imageW2x = imYCbCr.set0f(result).yCbCrToRgba()
		imageW2x.writeTo(outputVfs["kaifu2x.sample.png"], formats = PNG)

		val imageSideBySide = Bitmap32(image2x.width + imageW2x.width, image2x.height)
		imageSideBySide.put(image2x, 0, 0)
		imageSideBySide.put(imageW2x, image2x.width, 0)

		imageSideBySide.writeTo(outputVfs["kaifu2x.side2side.png"], formats = PNG)
		showImageAndWait(imageSideBySide)
	}
}

//fun Int.com.soywiz.kaifu2x.rgbaToYCbCr(): Int = TODO()
//fun Int.com.soywiz.kaifu2x.yCbCrToRgba(): Int = TODO()

fun Int.rgbaToYCbCr(): Int {
	val R = RGBA.getR(this)
	val G = RGBA.getG(this)
	val B = RGBA.getB(this)
	val A = RGBA.getA(this)

	val Y = (0 + (0.299 * R) + (0.587 * G) + (0.114 * B)).toInt().clamp(0, 255)
	val Cb = (128 - (0.168736 * R) - (0.331264 * G) + (0.5 * B)).toInt().clamp(0, 255)
	val Cr = (128 + (0.5 * R) - (0.418688 * G) - (0.081312 * B)).toInt().clamp(0, 255)

	return RGBA.pack(Y, Cb, Cr, A)
}

fun Int.yCbCrToRgba(): Int {
	val Y = RGBA.getR(this)
	val Cb = RGBA.getG(this)
	val Cr = RGBA.getB(this)
	val A = RGBA.getA(this)

	val R = (Y + 1.402 * (Cr - 128)).toInt().clamp(0, 255)
	val G = (Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)).toInt().clamp(0, 255)
	val B = (Y + 1.772 * (Cb - 128)).toInt().clamp(0, 255)

	return RGBA.pack(R, G, B, A)
}

fun Bitmap32.rgbaToYCbCr(): Bitmap32 = Bitmap32(width, height).apply { for (n in 0 until area) this.data[n] = this@rgbaToYCbCr.data[n].rgbaToYCbCr() }
fun Bitmap32.yCbCrToRgba(): Bitmap32 = Bitmap32(width, height).apply { for (n in 0 until area) this.data[n] = this@yCbCrToRgba.data[n].yCbCrToRgba() }

enum class ColorComponent(val index: Int) {
	RED(0), GREEN(1), BLUE(2), ALPHA(3);

	val shift = index * 8
	val clearMask = (0xFF shl shift).inv()

	fun extract(rgba: Int): Int = (rgba ushr shift) and 0xFF
	fun insert(rgba: Int, value: Int): Int = (rgba and clearMask) or ((value and 0xFF) shl shift)

	companion object {
		val ALL = values()
		operator fun get(index: Int) = ALL[index]
	}
}

fun Bitmap32.writeComponent(dstCmp: ColorComponent, from: Bitmap32, srcCmp: ColorComponent) {
	val fdata = from.data
	for (n in 0 until area) {
		data[n] = dstCmp.insert(data[n], srcCmp.extract(fdata[n]))
	}
}

fun Bitmap32.writeComponentf(dstCmp: ColorComponent, from: FloatArray2) = this.apply {
	val fdata = from.data
	for (n in 0 until area) {
		data[n] = dstCmp.insert(data[n], (fdata[n].clamp(0f, 1f) * 255).toInt())
	}
}

fun Bitmap32.readComponentf(cmp: ColorComponent, dst: FloatArray2 = FloatArray2(width, height)): FloatArray2 {
	val src = this
	val sdata = this.data
	val ddata = dst.data
	for (n in 0 until area) {
		ddata[n] = cmp.extract(sdata[n]).toFloat() / 255f
	}
	return dst
}

fun Bitmap32.get0f(): FloatArray2 = readComponentf(ColorComponent.RED)
fun Bitmap32.get1f(): FloatArray2 = readComponentf(ColorComponent.GREEN)
fun Bitmap32.get2f(): FloatArray2 = readComponentf(ColorComponent.BLUE)
fun Bitmap32.get3f(): FloatArray2 = readComponentf(ColorComponent.ALPHA)

fun Bitmap32.set0f(f: FloatArray2): Bitmap32 = writeComponentf(ColorComponent.RED, f)

fun Bitmap32.scaleNearest(sx: Int, sy: Int): Bitmap32 {
	val out = Bitmap32(width * sx, height * sy)
	for (y in 0 until out.height) {
		for (x in 0 until out.width) {
			out[x, y] = this[x / sx, y / sy]
		}
	}
	return out
}

fun Model.waifu2x(map: FloatArray2, parallel: Boolean = true, progressReport: (Int, Int) -> Unit = { cur, total -> }): FloatArray2 {
	var i_planes = arrayOf(map.paddedEdge(steps.size))
	val total = steps.map { it.nInputPlane * it.nOutputPlane }.sum()
	var current = 0
	val nthreads = if (parallel) Runtime.getRuntime().availableProcessors() else 1
	val tpool = Executors.newFixedThreadPool(nthreads)

	System.err.println("Processing Threads: $nthreads")

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

data class Model(val steps: List<Step>)

data class Step(
	var nInputPlane: Int = 1,
	var nOutputPlane: Int = 32,
	var dW: Int = 1,
	var dH: Int = 1,
	var kW: Int = 3,
	var kH: Int = 3,
	var padW: Int = 0,
	var padH: Int = 0,
	var model_config: List<ModelConfig> = listOf(ModelConfig()),
	var weight: List<List<FloatArray>> = listOf(listOf(FloatArray(9))),
	var bias: FloatArray = FloatArray(1),
	var class_name: String = "nn.SpatialConvolutionMM"
)

data class ModelConfig(
	var arch_name: String = "vgg_7",
	var scale_factor: Float = 1f,
	var channels: Int = 1,
	var offset: Int = 7
)

suspend fun getModel(): Model {
	System.err.print("Reading model...")
	val jsonString = resourcesVfs["models/scale2.0x_model.json"].readBytes().toString(ASCII)
	val json = Json.decode(jsonString)
	return parseModel(json).apply {
		System.err.println("Ok")
	}
}

fun parseModel(json: Any?): Model {
	return DynamicAccess {
		Model(json.list.map {
			Step(
				nInputPlane = it["nInputPlane"].int,
				nOutputPlane = it["nOutputPlane"].int,
				dW = it["dW"].int,
				dH = it["dH"].int,
				kW = it["kW"].int,
				kH = it["kH"].int,
				padW = it["padW"].int,
				padH = it["padH"].int,
				model_config = it["model_config"].list.map {
					ModelConfig(
						arch_name = it["arch_name"].str,
						scale_factor = it["scale_factor"].float,
						channels = it["channels"].int,
						offset = it["offset"].int
					)
				},
				weight = it["weight"].list.map {
					it.list.map {
						it.list.map { it.list }.flatMap { it }.floatArray
					}
				},
				bias = it["bias"].floatArray,
				class_name = it["class_name"].str
			)
		})
	}
}

fun sum(rr: Array<FloatArray2>): FloatArray2 = FloatArray2(rr[0].width, rr[0].height).setToAdd(rr)
fun min(l: FloatArray2, r: Float) = FloatArray2(l.width, l.height).apply { setToMin(l, r) }
fun max(l: FloatArray2, r: Float) = FloatArray2(l.width, l.height).apply { setToMax(l, r) }
fun clamp(l: FloatArray2, min: Float, max: Float) = FloatArray2(l.width, l.height).apply { setToClamp(l, min, max) }

class FloatArray2(val width: Int, val height: Int, val data: FloatArray = FloatArray(width * height)) {
	val area = width * height

	fun copy() = FloatArray2(width, height, data.copyOf())

	fun setToAdd(rr: Array<FloatArray2>) = this.apply {
		for (r in rr) {
			val rData = r.data
			for (n in 0 until area) data[n] += rData[n]
		}
	}

	fun setTo(r: FloatArray2) = setToFunc(r) { it }
	fun setToAdd(l: FloatArray2, r: FloatArray2) = setToFunc2(l, r) { l, r -> l + r }
	fun setToAdd(l: FloatArray2, r: Float) = setToFunc(l) { it + r }
	fun setToMul(l: FloatArray2, r: Float) = setToFunc(l) { it * r }
	fun setToMin(l: FloatArray2, r: Float) = setToFunc(l) { min(it, r) }
	fun setToMax(l: FloatArray2, r: Float) = setToFunc(l) { max(it, r) }
	fun setToClamp(l: FloatArray2, min: Float, max: Float) = setToFunc(l) { it.clamp(min, max) }

	operator fun plus(that: FloatArray2) = copy().apply { setToAdd(this, that) }
	operator fun plus(that: Float) = copy().apply { setToAdd(this, that) }
	operator fun times(that: Float) = copy().apply { setToMul(this, that) }
	operator fun div(that: Float) = copy().apply { setToMul(this, 1f / that) }

	fun index(x: Int, y: Int) = y * width + x
	operator fun get(x: Int, y: Int): Float = data[index(x, y)]
	operator fun set(x: Int, y: Int, value: Float): Unit = run { data[index(x, y)] = value }

	fun paddedEdge(pad: Int): FloatArray2 {
		val out = FloatArray2(width + pad * 2, height + pad * 2)
		for (y in 0 until height) {
			arraycopy(this.data, index(0, y), out.data, out.index(pad, y + pad), width)
		}
		return out
	}

	fun convolvedValidOptimized(f: FloatArray) = convolvedValidOptimized(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])
	fun convolvedValidUnoptimized(f: FloatArray) = convolvedValidUnoptimized(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])

	fun convolvedValidOptimized(
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) = FloatArray2(width - 2, height - 2).apply {
		setToConvolvedValidOptimized(this@FloatArray2, a, b, c, d, e, f, g, h, i)
	}

	fun convolvedValidUnoptimized(
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) = FloatArray2(width - 2, height - 2).apply {
		setToConvolvedValidUnoptimized(this@FloatArray2, a, b, c, d, e, f, g, h, i)
	}

	fun setToConvolvedValidOptimized(src: FloatArray2, f: FloatArray) =
		setToConvolvedValidOptimized(src, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])

	// Optimize!
	// 24 seconds
	fun setToConvolvedValidOptimized(
		src: FloatArray2,
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) {
		val dst = this
		if ((dst.width != src.width - 2) || (dst.height != src.height - 2)) {
			invalidArg("Invalid image sizes")
		}
		val dstData = dst.data
		val srcData = src.data
		val srcWidth = src.width

		for (y in 0 until dst.height) {
			var sp = src.index(1, y + 1)
			var dp = dst.index(0, y)

			var _a = srcData[sp - srcWidth - 1]
			var _b = srcData[sp - srcWidth + 0]

			var _d = srcData[sp - 1]
			var _e = srcData[sp + 0]

			var _g = srcData[sp + srcWidth - 1]
			var _h = srcData[sp + srcWidth + 0]

			for (x in 0 until dst.width) {
				val _c = srcData[sp - srcWidth + 1]
				val _f = srcData[sp + 1]
				val _i = srcData[sp + srcWidth + 1]

				dstData[dp] = 0f +
					a * _a +
					b * _b +
					c * _c +
					d * _d +
					e * _e +
					f * _f +
					g * _g +
					h * _h +
					i * _i

				// Shift
				_a = _b
				_b = _c

				_d = _e
				_e = _f

				_g = _h
				_h = _i

				sp++
				dp++
			}
		}
	}

	// 30 seconds
	fun setToConvolvedValidUnoptimized(
		src: FloatArray2,
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) {
		val dst = this
		assert(dst.width == src.width - 2)
		assert(dst.height == src.height - 2)
		val dstData = dst.data
		val srcData = src.data
		val srcWidth = src.width

		// @TODO: THREADS --> Probably better at other level
		for (y in 0 until dst.height) {
			var sp = src.index(1, y + 1)
			var dp = dst.index(0, y)

			// @TODO: SIMD
			// @TODO: Reduce reads by keeping a sliding window (9 memory reads for step -> 3 memory reads for step)
			for (x in 0 until dst.width) {
				val _a = srcData[sp - srcWidth - 1]
				val _b = srcData[sp - srcWidth + 0]
				val _c = srcData[sp - srcWidth + 1]

				val _d = srcData[sp - 1]
				val _e = srcData[sp + 0]
				val _f = srcData[sp + 1]

				val _g = srcData[sp + srcWidth - 1]
				val _h = srcData[sp + srcWidth + 0]
				val _i = srcData[sp + srcWidth + 1]

				dstData[dp] = 0f +
					a * _a +
					b * _b +
					c * _c +
					d * _d +
					e * _e +
					f * _f +
					g * _g +
					h * _h +
					i * _i

				sp++
				dp++
			}
		}
	}

	inline fun setToFunc(r: FloatArray2, func: (Float) -> Float) = this.apply {
		val tdata = this.data
		val rdata = r.data
		for (n in 0 until area) tdata[n] = func(rdata[n])
	}

	inline fun setToFunc2(l: FloatArray2, r: FloatArray2, func: (Float, Float) -> Float) = this.apply {
		val tdata = this.data
		val ldata = l.data
		val rdata = r.data
		for (n in 0 until area) tdata[n] = func(ldata[n], rdata[n])
	}

	fun areAllEqualTo(v: Float): Boolean {
		for (n in 0 until area) if (data[n] != v) return false
		return true
	}

	override fun toString(): String = "com.soywiz.kaifu2x.FloatArray2($width, $height)"
}
