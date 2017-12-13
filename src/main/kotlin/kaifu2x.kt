import com.soywiz.kmem.arraycopy
import com.soywiz.korim.bitmap.Bitmap
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korim.format.PNG
import com.soywiz.korim.format.writeTo
import com.soywiz.korio.Korio
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.UTF8
import com.soywiz.korio.lang.toByteArray
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.util.clamp
import com.soywiz.korio.vfs.LocalVfs
import com.soywiz.korio.vfs.resourcesVfs
import org.tensorflow.TensorFlow
import kotlin.math.max
import kotlin.math.min


fun tfdemo() {
	graph {
		val value = "Hello from " + TensorFlow.version()

		val const1 = value.toByteArray(UTF8).u8ArrayConstant()
		val const2 = (0 until 16).map { 1.toByte() }.toByteArray().u8ArrayConstant()

		session {
			println(const1.getBytes().toString(UTF8))
			println(const2.getBytes().toString(UTF8))
			println((const1 + const2).getBytes().toString(UTF8))
		}
	}

	//Graph().auto { g ->
	//	val value = "Hello from " + TensorFlow.version()
//
	//	// Construct the computation graph with a single operation, a constant
	//	// named "MyConst" with a value "value".
	//	Tensor.create(value.toByteArray(UTF8)).auto { t ->
	//		// The Java API doesn't yet include convenience functions for adding operations.
	//		g.opBuilder("Const", "MyConst")
	//			.setAttr("dtype", t.dataType())
	//			.setAttr("value", t)
	//			.build()
	//	}
//
	//	// Execute the "MyConst" operation in a Session.
	//	Session(g).auto { s ->
	//		s.runner().fetch("MyConst").run().get(0).auto { output ->
	//			println(output.bytesValue().toString(UTF8))
	//		}
	//	}
	//}
}

fun main(args: Array<String>) = Korio {
	tfdemo()

	val model = getModel()
	val image = PNG.decode(resourcesVfs["samples/small.png"]).toBMP32()
	val im = image.scaleNearest(2, 2)
	val imYCbCr = im.rgbaToYCbCr()
	val pad = model.steps.size
	val paddedY = imYCbCr.get0f().paddedEdge(pad)
	val result = model.waifu2xInternal(paddedY) { current, total ->
		println("$current/$total")
	}
	val out: Bitmap = imYCbCr.set0f(result).yCbCrToRgba()
	out.writeTo(LocalVfs("/tmp/kaifu2x.sample.png"), formats = PNG)
	println(paddedY)
	println(result.unpaddedEdge(pad))
}

//fun Int.rgbaToYCbCr(): Int = TODO()
//fun Int.yCbCrToRgba(): Int = TODO()

fun Int.rgbaToYCbCr(): Int {
	val r = RGBA.getR(this)
	val g = RGBA.getG(this)
	val b = RGBA.getB(this)
	val a = RGBA.getA(this)

	val Y = (0 + (0.299 * r) + (0.587 * g) + (0.114 * b)).toInt().clamp(0, 255)
	val Cb = (128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)).toInt().clamp(0, 255)
	val Cr = (128 + (0.5 * r) - (0.418688 * g) + (0.081312 * b)).toInt().clamp(0, 255)

	return RGBA.pack(Y, Cb, Cr, a)
}

fun Int.yCbCrToRgba(): Int {
	val Y = RGBA.getR(this)
	val Cb = RGBA.getG(this)
	val Cr = RGBA.getB(this)
	val a = RGBA.getA(this)

	val R = (Y + 1.402 * (Cr - 128)).toInt().clamp(0, 255)
	val G = (Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)).toInt().clamp(0, 255)
	val B = (Y + 1.772 * (Cb - 128)).toInt().clamp(0, 255)

	return RGBA.pack(R, G, B, a)
}

fun Bitmap32.rgbaToYCbCr(): Bitmap32 = Bitmap32(width, height).apply { for (n in 0 until area) this.data[n] = this@rgbaToYCbCr.data[n].rgbaToYCbCr() }
fun Bitmap32.yCbCrToRgba(): Bitmap32 = Bitmap32(width, height).apply { for (n in 0 until area) this.data[n] = this@yCbCrToRgba.data[n].yCbCrToRgba() }

fun Bitmap32.get0f(): FloatArray2 = FloatArray2(width, height).apply { for (n in 0 until area) this.data[n] = RGBA.getRf(this@get0f.data[n]) }
fun Bitmap32.get1f(): FloatArray2 = FloatArray2(width, height).apply { for (n in 0 until area) this.data[n] = RGBA.getGf(this@get1f.data[n]) }
fun Bitmap32.get2f(): FloatArray2 = FloatArray2(width, height).apply { for (n in 0 until area) this.data[n] = RGBA.getBf(this@get2f.data[n]) }

fun Bitmap32.set0f(f: FloatArray2): Bitmap32 = this.apply {
	for (n in 0 until area) this.data[n] = (this.data[n] and 0x000000FF.inv()) or (f.data[n].clamp(0f, 1f) * 255).toInt()
}

fun Bitmap32.scaleNearest(sx: Int, sy: Int): Bitmap32 {
	val out = Bitmap32(width * sx, height * sy)
	for (y in 0 until out.height) {
		for (x in 0 until out.width) {
			out[x, y] = this[x / sx, y / sy]
		}
	}
	return out
}

fun Model.waifu2xInternal(paddedMap: FloatArray2, progressReport: (Int, Int) -> Unit = { cur, total -> }): FloatArray2 {
	var planes = listOf(paddedMap)
	val total = steps.map { it.nInputPlane * it.nOutputPlane }.sum()
	var current = 0
	progressReport(0, total)
	for (step in steps) {
		val o_planes = arrayListOf<FloatArray2>()
		for ((bias, weights) in step.bias.zip(step.weight)) {
			var partial: FloatArray2? = null
			for ((ip, kernel) in planes.zip(weights)) {
				val p = ip.convolvedValid(kernel)
				if (partial == null) {
					partial = p
				} else {
					partial += p
				}
				current++
				progressReport(current, total)
			}
			o_planes += partial!! + bias
		}
		planes = o_planes.map { p -> max(p, 0f) + min(p, 0f) * 0.1f }
	}
	return planes.first()
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
	println("Start")
	val jsonString = resourcesVfs["models/scale2.0x_model.json"].readBytes().toString(ASCII)
	println("Readed Json")
	val json = Json.decode(jsonString)
	val model = parseModel(json)
	println("Decoded Json")
	return model
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

fun min(l: FloatArray2, r: Float) = FloatArray2(l.width, l.height).apply { setToMin(l, r) }
fun max(l: FloatArray2, r: Float) = FloatArray2(l.width, l.height).apply { setToMax(l, r) }
fun clamp(l: FloatArray2, min: Float, max: Float) = FloatArray2(l.width, l.height).apply { setToClamp(l, min, max) }

class FloatArray2(val width: Int, val height: Int, val data: FloatArray = FloatArray(width * height)) {
	val area = width * height

	fun copy() = FloatArray2(width, height, data.copyOf())

	fun setToAdd(l: FloatArray2, r: FloatArray2) = run { for (n in 0 until area) data[n] = l.data[n] + r.data[n] }
	fun setToAdd(l: FloatArray2, r: Float) = run { for (n in 0 until area) data[n] = l.data[n] + r }
	fun setToMul(l: FloatArray2, r: Float) = run { for (n in 0 until area) data[n] = l.data[n] * r }
	fun setToMin(l: FloatArray2, r: Float) = run { for (n in 0 until area) data[n] = min(l.data[n], r) }
	fun setToMax(l: FloatArray2, r: Float) = run { for (n in 0 until area) data[n] = max(l.data[n], r) }
	fun setToClamp(l: FloatArray2, min: Float, max: Float) = run { for (n in 0 until area) data[n] = min(max(l.data[n], min), max) }

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

	fun unpaddedEdge(pad: Int): FloatArray2 {
		val out = FloatArray2(width - pad * 2, height - pad * 2)
		println("$this -> $out")
		for (y in 0 until out.height) {
			arraycopy(this.data, index(pad, y + pad), out.data, out.index(0, y), out.width)
		}
		return out
	}

	fun convolvedValid(f: FloatArray) = convolvedValid(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])

	fun convolvedValid(
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) = copy().apply {
		setToConvolvedValid(this@FloatArray2, a, b, c, d, e, f, g, h, i)
	}

	// Optimize!
	fun setToConvolvedValid(
		src: FloatArray2,
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) {
		assert(width == src.width)
		assert(height == src.height)
		val srcData = src.data

		// @TODO: THREADS
		for (y in 1 until height - 1) {
			var n = index(1, y)

			// @TODO: SIMD
			// @TODO: Reduce reads by keeping a sliding window (9 memory reads for step -> 3 memory reads for step)
			for (x in 1 until width - 1) {
				val _a = srcData[n - width - 1]
				val _b = srcData[n - width + 0]
				val _c = srcData[n - width + 1]

				val _d = srcData[n - 1]
				val _e = srcData[n + 0]
				val _f = srcData[n + 1]

				val _g = srcData[n + width - 1]
				val _h = srcData[n + width + 0]
				val _i = srcData[n + width + 1]

				this.data[n] = 0f +
					a * _a +
					b * _b +
					c * _c +
					d * _d +
					e * _e +
					f * _f +
					g * _g +
					h * _h +
					i * _i

				n++
			}
		}
	}

	override fun toString(): String = "FloatArray2($width, $height)"
}
