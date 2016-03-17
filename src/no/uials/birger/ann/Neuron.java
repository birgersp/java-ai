package no.uials.birger.ann;

import java.util.Random;

public class Neuron {

	private final double beta;
	private final double[] weights;

	public Neuron(int inputs) {

		this(inputs, 1);

	}

	public Neuron(double... weights) {

		this.weights = weights;
		this.beta = 1;

	}

	public Neuron(int inputs, double beta) {

		Random r = new Random(System.currentTimeMillis());

		double sqrtN = Math.sqrt(inputs);
		weights = new double[inputs + 1];
		for (int i = 0; i < weights.length; i++)
			weights[i] = 1 / sqrtN * (2 * r.nextDouble() - 1);

		this.beta = beta;

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

		return sigmoid(beta, sum);

	}

	public static double sigmoid(double beta, double value) {

		return 1.0 / (1.0 + Math.exp(beta * -value));

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
