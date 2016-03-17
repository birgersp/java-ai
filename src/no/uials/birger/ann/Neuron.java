package no.uials.birger.ann;

import java.util.Random;
import java.util.function.DoubleFunction;

public class Neuron {

	public static Neuron getRandom(DoubleFunction<Double> f, int inputs) {

		Random r = new Random(System.currentTimeMillis());

		double sqrtN = Math.sqrt(inputs);
		double[] weights = new double[inputs + 1];
		for (int i = 0; i < weights.length; i++)
			weights[i] = 1 / sqrtN * (2 * r.nextDouble() - 1);
		return new Neuron(f, weights);

	}

	private final DoubleFunction<Double> f;
	private final double[] weights;

	public Neuron(DoubleFunction<Double> f, double... weights) {

		this.f = f;
		this.weights = weights;

	}

	public double[] getWeights() {

		return weights;

	}

	public int noOfInputs() {

		return weights.length - 1;

	}

	public double recall(double[] input) {

		double sum = weights[0] * -1;
		for (int i = 0; i < input.length; i++) {
			sum += weights[i + 1] * input[i];
		}

		return f.apply(sum);

	}

	public void train(double[] input, double expectation, double learningRate) {

		double result = recall(input);
		if (expectation != result) {

			double x = -1;
			for (int i = -1; i < input.length; i++) {
				if (i >= 0)
					x = input[i];

				weights[i + 1] -= learningRate * (result - expectation) * x;
			}

		}

	}

}
