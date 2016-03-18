package no.uials.birger.ann;

import java.util.Random;

public class Neuron {

	public static Neuron getRandom(int inputs) {

		Random r = new Random(System.currentTimeMillis());

		double sqrtN = Math.sqrt(inputs);
		double[] weights = new double[inputs + 1];
		for (int i = 0; i < weights.length; i++)
			weights[i] = 1 / sqrtN * (2 * r.nextDouble() - 1);
		return new Neuron(weights);

	}

	private double[] weights;

	public Neuron(double... weights) {

		this.weights = weights;

	}

	public double[] getWeights() {

		return weights;

	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}
	
}
