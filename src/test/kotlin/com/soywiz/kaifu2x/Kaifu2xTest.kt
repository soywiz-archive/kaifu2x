package com.soywiz.kaifu2x

import com.soywiz.korim.bitmap.A
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korim.bitmap.Y
import com.soywiz.korim.format.*
import com.soywiz.korio.async.syncTest
import com.soywiz.korio.crypto.toBase64
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.vfs.VfsFile
import com.soywiz.korio.vfs.resourcesVfs
import org.junit.Test
import kotlin.math.log10
import kotlin.math.sqrt
import kotlin.test.assertTrue

class Kaifu2xTest {
	init {
		defaultImageFormats.registerStandard()
	}

	private fun testScale2x(input: VfsFile, expected: VfsFile, vararg channels: BitmapChannel, psnrMin: Double = 45.0) = syncTest {
		val inputBmp = input.readBitmapNoNative().toBMP32()
		val expectedBmp = expected.readBitmapNoNative().toBMP32()
		val result = Kaifu2x.scaleRgba(inputBmp, 2, channels = channels.toList(), output = null)
		val psnr = PSNR(expectedBmp, result)
		// 0.5 dB increments are considered noticeable
		if (psnr < psnrMin) {
			//println("Expected:" + PNG.encode(expectedBmp).toBase64())
			//println("Result:" + PNG.encode(result).toBase64())
		}
		assertTrue(psnr >= psnrMin, "PSNR was $psnr but we need it to be $psnrMin dB or better")

	}

	@Test
	fun testScale2xYCbCrA() {
		testScale2x(
			resourcesVfs["goku_small_bg.png"],
			resourcesVfs["goku_small_bg.2x.ycbcra.png"],
			*BitmapChannel.ALL
		)
	}

	@Test
	fun testScale2xYA() {
		testScale2x(
			resourcesVfs["goku_small_bg.png"],
			resourcesVfs["goku_small_bg.2x.ya.png"],
			BitmapChannel.Y, BitmapChannel.A
		)
	}

	@Test
	fun testScale2xYA2() {
		testScale2x(
			resourcesVfs["goku_small_bg.png"],
			resourcesVfs["goku_small_bg.2x.ya.png"],
			BitmapChannel.Y, BitmapChannel.A
		)
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
