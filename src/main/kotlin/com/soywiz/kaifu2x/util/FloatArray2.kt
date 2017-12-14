package com.soywiz.kaifu2x.util

import com.soywiz.kmem.arraycopy
import com.soywiz.korio.error.invalidArg
import com.soywiz.korio.util.clamp

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
	fun setToMin(l: FloatArray2, r: Float) = setToFunc(l) { kotlin.math.min(it, r) }
	fun setToMax(l: FloatArray2, r: Float) = setToFunc(l) { kotlin.math.max(it, r) }
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

	override fun toString(): String = "com.soywiz.kaifu2x.util.FloatArray2($width, $height)"
}

fun sum(rr: Array<FloatArray2>): FloatArray2 = FloatArray2(rr[0].width, rr[0].height).setToAdd(rr)
fun min(l: FloatArray2, r: Float) = FloatArray2(l.width, l.height).apply { setToMin(l, r) }
fun max(l: FloatArray2, r: Float) = FloatArray2(l.width, l.height).apply { setToMax(l, r) }
fun clamp(l: FloatArray2, min: Float, max: Float) = FloatArray2(l.width, l.height).apply { setToClamp(l, min, max) }
