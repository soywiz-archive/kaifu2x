import org.tensorflow.*
import org.tensorflow.types.UInt8

// https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java

inline fun <T : AutoCloseable, TR> T.auto(callback: (T) -> TR): TR {
	try {
		return callback(this)
	} finally {
		this.close()
	}
}

//fun <T> Output<T>.
fun Graph.binaryOp(type: String, l: Output<*>, r: Output<*>): Output<*> {
	return opBuilder(type, type).addInput(l).addInput(r).build().output<Any>(0)
}

fun Graph.div(x: Output<*>, y: Output<*>): Output<*> = binaryOp("Div", x, y)
fun Graph.mul(x: Output<*>, y: Output<*>): Output<*> = binaryOp("Mul", x, y)
fun Graph.sub(x: Output<*>, y: Output<*>): Output<*> = binaryOp("Sub", x, y)
fun Graph.add(x: Output<*>, y: Output<*>): Output<*> = binaryOp("Add", x, y)

interface IOutput<T> {
	val output: Output<T>
}
data class NamedOutput<T>(val name: String, override val output: Output<T>) : IOutput<T> {
	constructor(name: String, output: IOutput<T>) : this(name, output.output)
}
data class UnnamedOutput<T>(override val output: Output<T>) : IOutput<T>
fun <T> IOutput<T>.named(name: String) = NamedOutput(name, this.output)

class WrappedSession(val session: Session) {
	fun IOutput<*>.getBytes(): ByteArray {
		val tensor = session.runner().fetch(this.output.op().name()).run().first()
		val out = ByteArray(tensor.numBytes())
		tensor.copyTo(out)
		return out
	}
}

class GraphBuilder(val g: Graph) {
	var lastId: Int = 0
	operator fun IOutput<*>.div(that: IOutput<*>): IOutput<*> = UnnamedOutput(g.div(this.output, that.output))
	operator fun IOutput<*>.times(that: IOutput<*>): IOutput<*> = UnnamedOutput(g.mul(this.output, that.output))
	operator fun IOutput<*>.plus(that: IOutput<*>): IOutput<*> = UnnamedOutput(g.add(this.output, that.output))
	operator fun IOutput<*>.minus(that: IOutput<*>): IOutput<*> = UnnamedOutput(g.sub(this.output, that.output))

	fun <T> constant(name: String, t: Tensor<T>): Operation {
		return g.opBuilder("Const", name).setAttr("dtype", t.dataType()).setAttr("value", t).build()
	}

	inline fun <TR> tensor(value: ByteArray, callback: (Tensor<*>) -> TR): TR {
		return Tensor.create(value).auto { t -> callback(t) }
	}

	inline fun <TR> session(callback: WrappedSession.() -> TR): TR {
		return Session(g).auto { callback(WrappedSession(it)) }
	}

	//inline fun <TR> Session.runner(callback: SessionRunner.() -> TR): TR {
	//	return callback(SessionRunner(runner()))
	//}

	fun <T> constant(name: String, value: Any, type: Class<T>, dtype: DataType = DataType.fromClass(type)): UnnamedOutput<T> {
		return Tensor.create(value, type).auto { t ->
			UnnamedOutput<T>(g.opBuilder("Const", name)
				.setAttr("dtype", dtype)
				.setAttr("value", t)
				.build()
				.output(0))
		}
	}

	fun Session.Runner.fetch(named: IOutput<*>) = this.fetch(named.output.op().name())
	fun Session.Runner.fetchRun(named: IOutput<*>) = fetch(named).run()
	fun Session.Runner.fetchRunFirst(named: IOutput<*>) = fetchRun(named).first()

	fun allocName() = "Auto${lastId++}"

	fun ByteArray.u8ArrayConstant() = allocName().run { constant(this, this@u8ArrayConstant, UInt8::class.java).named(this) }
	fun ByteArray.stringConstant() = allocName().run { constant(this, this@stringConstant, String::class.java).named(this) }

	fun constant(value: ByteArray): NamedOutput<String> = allocName().run { constant(this, value).named(this) }
	fun constant(value: Int): NamedOutput<Int> = allocName().run { constant(this, value).named(this) }
	fun constant(value: Float): NamedOutput<Float> = allocName().run { constant(this, value).named(this) }
	fun constant(value: IntArray): NamedOutput<Int> = allocName().run { constant(this, value).named(this) }
	fun constant(value: FloatArray): NamedOutput<Float> = allocName().run { constant(this, value).named(this) }

	fun constant(name: String, value: ByteArray): NamedOutput<String> = this.constant(name, value, String::class.java).named(name)

	fun constant(name: String, value: Int): NamedOutput<Int> = this.constant(name, value, Int::class.java).named(name)
	fun constant(name: String, value: IntArray): NamedOutput<Int> = this.constant(name, value, Int::class.java).named(name)

	fun constant(name: String, value: Float): NamedOutput<Float> = this.constant(name, value, Float::class.java).named(name)
	fun constant(name: String, value: FloatArray): NamedOutput<Float> = this.constant(name, value, Float::class.java).named(name)
}

inline fun <T> Graph.builder(callback: GraphBuilder.() -> T): T {
	return callback(GraphBuilder(this))
}

inline fun <T> graph(callback: GraphBuilder.() -> T): T {
	return Graph().auto { g -> callback(GraphBuilder(g)) }
}
