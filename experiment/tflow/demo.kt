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
}
