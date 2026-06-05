import _ from "lodash";
import * as fs from "fs";

const names = fs.readFileSync("names.txt", "utf8");
const docs = _.shuffle(
	names
		.split("\n")
		.map((name) => name.trim())
		.filter((name) => name.length > 0),
);

console.log(`num docs: ${docs.length}`);

const uchars = [...new Set<string>([...docs.join("")])].sort();
const BOS = uchars.length;
const vocabSize = uchars.length + 1;

console.log(`vocab size: ${vocabSize}`);

class Value {
	public grad: number = 0;

	constructor(
		public readonly data: number,
		public readonly children: [Value, Value] | [Value] | null = null,
		public readonly localGrads: [number, number] | [number] | null = null,
	) {}

	public add(other: Value | number): Value {
		const _other = typeof other === "number" ? new Value(other) : other;
		return new Value(this.data + _other.data, [this, _other], [1, 1]);
	}

	public mul(other: Value | number): Value {
		const _other = typeof other === "number" ? new Value(other) : other;
		return new Value(
			this.data * _other.data,
			[this, _other],
			[_other.data, this.data],
		);
	}

	public pow(other: Value | number): Value {
		const _other = typeof other === "number" ? new Value(other) : other;
		return new Value(
			Math.pow(this.data, _other.data),
			[this],
			[_other.data * Math.pow(this.data, _other.data - 1)],
		);
	}

	public log(): Value {
		return new Value(Math.log(this.data), [this], [1 / this.data]);
	}

	public exp(): Value {
		return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
	}

	public relu(): Value {
		return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
	}

	public neg(): Value {
		return this.mul(-1);
	}

	public sub(other: Value | number): Value {
		const _other = typeof other === "number" ? new Value(other) : other;
		return this.add(_other.neg());
	}

	public div(other: Value | number): Value {
		const _other = typeof other === "number" ? new Value(other) : other;
		return this.mul(_other.pow(-1));
	}

	public backward(): void {
		const topo: Value[] = [];
		const visited = new Set<Value>();

		const buildTopo = (v: Value) => {
			if (visited.has(v)) return;
			visited.add(v);
			if (v.children) {
				for (const child of v.children) {
					if (child === null) continue;
					buildTopo(child);
				}
				topo.push(v);
			}
		};

		buildTopo(this);
		this.grad = 1;

        // why are we reversing the topo?
		for (const v of topo.reverse()) {
			if (v.children === null) continue;
			if (v.localGrads === null) continue;
			const zipped = _.zip(v.children, v.localGrads);

			for (const [child, localGrad] of zipped) {
				if (child === undefined) continue;
				if (localGrad === undefined) continue;
				child.grad += localGrad * v.grad;
			}
		}
	}
}
