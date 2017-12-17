package com.soywiz.kaifu2x

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korim.format.defaultImageFormats
import com.soywiz.korim.format.readBitmap
import com.soywiz.korim.format.registerStandard
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.vfs.resourcesVfs
import org.junit.Test
import kotlin.math.log10
import kotlin.math.sqrt
import kotlin.test.assertTrue

class Kaifu2xTest {
	init {
		defaultImageFormats.registerStandard()
	}

	@Test
	fun testScale2x() = syncTest {
		val input = resourcesVfs["goku_small_bg.png"].readBitmap().toBMP32()
		val expected = resourcesVfs["goku_small_bg.2x.png"].readBitmap().toBMP32()
		val result = Kaifu2x.scaleRgba(input, 2, channels = BitmapChannel.ALL.toList())
		val psnr = PSNR(expected, result)
		val psnrMin = 50.0
		// 0.5 dB increments are considered noticeable
		assertTrue(psnr >= psnrMin, "PSNR was $psnr but we need it to be $psnrMin dB or better")
	}
}

// https://en.wikipedia.org/wiki/PSNR
// @TODO: Move to Korim
object PSNR {
	fun MSE(a: Bitmap32, b: Bitmap32, c: BitmapChannel): Double {
		if (a.size != b.size) invalidOp("${a.size} != ${b.size}")
		val area = a.area
		var sum = 0.0
		for (n in 0 until area) {
			val v = c.extract(a.data[n]) - c.extract(b.data[n])
			sum += v * v
		}
		return sum / area.toDouble()
	}

	fun MSE(a: Bitmap32, b: Bitmap32): Double {
		return BitmapChannel.ALL.map { MSE(a, b, it) }.sum() / 4.0
	}

	private fun PSNR(a: Bitmap32, b: Bitmap32, mse: Double): Double {
		return 20.0 * log10(0xFF.toDouble() / sqrt(mse))
	}

	operator fun invoke(a: Bitmap32, b: Bitmap32): Double = PSNR(a, b, MSE(a, b))
	operator fun invoke(a: Bitmap32, b: Bitmap32, c: BitmapChannel): Double = PSNR(a, b, MSE(a, b, c))
}
