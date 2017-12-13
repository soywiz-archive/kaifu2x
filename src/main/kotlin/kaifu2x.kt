import com.soywiz.korio.Korio
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.vfs.resourcesVfs
import kotlin.math.max
import kotlin.math.min

class ModelConfig {
	var arch_name = "vgg_7"
	var scale_factor = 1.0
	var channels = 1
	var offset = 7
}

class Weight(
	var a: Float = 0f, var b: Float = 0f, var c: Float = 0f,
	var d: Float = 0f, var e: Float = 0f, var f: Float = 0f,
	var g: Float = 0f, var h: Float = 0f, var i: Float = 0f
)

data class Step(
	var dW: Int = 1,
	var dH: Int = 1,
	var nInputPlane: Int = 1,
	var kW: Int = 3,
	var kH: Int = 3,
	var padW: Int = 0,
	var padH: Int = 0,
	var model_config: List<ModelConfig> = listOf(ModelConfig()),
	var weight: List<Weight> = listOf(Weight()),
	var bias: List<Float> = listOf(1f),
	var nOutputPlane: Int = 32,
	var class_name: String = "nn.SpatialConvolutionMM"
)

fun main(args: Array<String>) = Korio {
	println("Start")
	val jsonString = resourcesVfs["models/scale2.0x_model.json"].readBytes().toString(ASCII)
	println("Readed Json")
	val json = Json.decode(jsonString)
	println("Decoded Json")
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

	fun index(x: Int, y: Int) = y * width + x
	operator fun get(x: Int, y: Int): Float = data[index(x, y)]
	operator fun set(x: Int, y: Int, value: Float): Unit = run { data[index(x, y)] = value }

	fun convolved(
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) = copy().apply {
		setToConvolved(this@FloatArray2, a, b, c, d, e, f, g, h, i)
	}

	// @TODO: Optimize
	fun setToConvolved(
		src: FloatArray2,
		a: Float, b: Float, c: Float,
		d: Float, e: Float, f: Float,
		g: Float, h: Float, i: Float
	) {
		assert(width == src.width)
		assert(height == src.height)
		val srcData = src.data

		for (n in 0 until area) {
			val a0 = a * srcData.getOrElse(n - width - 1) { 0f }
			val b0 = b * srcData.getOrElse(n - width) { 0f }
			val c0 = c * srcData.getOrElse(n - width + 1) { 0f }

			val d0 = d * srcData.getOrElse(n - 1) { 0f }
			val e0 = e * srcData.getOrElse(n) { 0f }
			val f0 = f * srcData.getOrElse(n + 1) { 0f }

			val g0 = g * srcData.getOrElse(n + width - 1) { 0f }
			val h0 = h * srcData.getOrElse(n + width) { 0f }
			val i0 = i * srcData.getOrElse(n + width + 1) { 0f }

			this.data[n] = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0 + i0
		}
	}
}
