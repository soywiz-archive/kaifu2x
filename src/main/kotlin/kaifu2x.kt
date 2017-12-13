import com.soywiz.kmem.arraycopy
import com.soywiz.korio.Korio
import com.soywiz.korio.lang.ASCII
import com.soywiz.korio.lang.toString
import com.soywiz.korio.serialization.json.Json
import com.soywiz.korio.vfs.resourcesVfs
import kotlin.math.max
import kotlin.math.min

data class ModelConfig(
	var arch_name: String = "vgg_7",
	var scale_factor: Float = 1f,
	var channels: Int = 1,
	var offset: Int = 7
)

//data class Weight(var weights: FloatArray = FloatArray(9))
//data class Weights(val weights: List<Weight> = listOf())
data class Weights(var weights: FloatArray = FloatArray(9))
data class WeightsList(val weights: List<Weights> = listOf())

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
	var weight: List<WeightsList> = listOf(WeightsList()),
	var bias: FloatArray = FloatArray(1),
	var class_name: String = "nn.SpatialConvolutionMM"
)

data class Model(val steps: List<Step>)

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
					WeightsList(it.list.map {
						Weights(it.list.map { it.list }.flatMap { it }.floatArray)
					})
				},
				bias = it["bias"].floatArray,
				class_name = it["class_name"].str
			)
		})
	}
}

suspend fun getModel(): Model {
	println("Start")
	val jsonString = resourcesVfs["models/scale2.0x_model.json"].readBytes().toString(ASCII)
	println("Readed Json")
	val json = Json.decode(jsonString)

	val model = parseModel(json)
	println("Decoded Json")
	return model
}

fun main(args: Array<String>) = Korio {
	val model = getModel()
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

	fun paddedEdge(pad: Int): FloatArray2 {
		val out = FloatArray2(width + pad * 2, height + pad * 2)
		for (y in 0 until height) {
			arraycopy(this.data, index(0, y), out.data, index(pad, y), width)
		}
		return out
	}

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

		// THREADS
		for (y in 1 until height - 1) {
			var n = y * width + 1

			// SIMD/Reduce reads by keeping a sliding window
			for (x in 1 until width - 1) {
				val a0 = a * srcData[n - width - 1]
				val b0 = b * srcData[n - width + 0]
				val c0 = c * srcData[n - width + 1]

				val d0 = d * srcData[n - 1]
				val e0 = e * srcData[n + 0]
				val f0 = f * srcData[n + 1]

				val g0 = g * srcData[n + width - 1]
				val h0 = h * srcData[n + width + 0]
				val i0 = i * srcData[n + width + 1]

				this.data[n] = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0 + i0

				n++
			}
		}
	}
}
