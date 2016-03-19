package no.uials.birger.ann;

import java.util.Random;
import java.util.function.DoubleFunction;

public class Network {

	private static double[] getRandomNeuron(int inputs) {

		Random r = new Random(System.currentTimeMillis());

		double sqrtN = Math.sqrt(inputs);
		double[] w = new double[inputs + 1];
		for (int i = 0; i < w.length; i++)
			w[i] = 1 / sqrtN * (2 * r.nextDouble() - 1);

		return w;

	}

	public static Network getRandom(DoubleFunction<Double> f, DoubleFunction<Double> fDerivative, int... layerOutputs) {

		// Number of layers
		int L = layerOutputs.length - 1;

		// Create weights
		double[][][] w = new double[L][][];

		// Number of neurons for layer l
		int J;

		// For each layer l
		for (int l = 0; l < L; l++) {

			// Compute number of outputs (neurons) in layer l
			J = layerOutputs[l + 1];

			// Create neurons
			w[l] = new double[J][];

			// For each output
			for (int j = 0; j < J; j++)
				w[l][j] = getRandomNeuron(layerOutputs[l]);

		}

		return new Network(f, fDerivative, w);

	}

	private final DoubleFunction<Double> f;
	private final DoubleFunction<Double> fD;
	private final double[][][] w;
	private final int L;

	public Network(DoubleFunction<Double> f, DoubleFunction<Double> fDerivative, double[][][] w) {

		this.f = f;
		this.fD = fDerivative;
		this.w = w;
		this.L = w.length;

	}

	public double[] recall(double[] input) {

		// Inputs
		double[] x = input;

		// Outputs
		double[] y;

		// For each layer l
		for (int l = 0; l < L; l++) {

			y = recallLayer(l, x);
			x = y;

		}

		return x;

	}

	private double[] recallLayer(int l, double[] input) {

		// Number of outputs of layer l
		final int J = w[l].length;

		// Layer output
		double[] y = new double[J];

		// For each neuron (output j)
		for (int j = 0; j < J; j++) {

			// Apply function on neuron signal
			y[j] = f.apply(recallSignal(l, j, input));

		}

		return y;

	}

	private double recallSignal(int l, int j, double[] input) {

		// Compute neuron signal
		double s = w[l][j][0] * -1;

		// For each input i
		for (int i = 1; i < w.length; i++)
			s += w[l][j][i] * input[i - 1];

		return s;

	}

	public void train(double[] input, double[] target, double learningRate) {

		// Output of layer outputs: x[layer][output]
		double[][] x = new double[L][];
		
		// Signal of layer neruons: s[layer][output]
		double[][] s = new double[L][];

		// Forward run
		// Compute output of each layer
		
		x[0] = input;
		for (int l = 1; l < L; l++) {
			
			final int J = w[l].length;
			
			// For each neuron (output j)
			for (int j = 0; j < J; j++) {

				// Compute neuron signal
				s[l][j] = recallSignal(l, j, x[l-1]);
				
				// Apply function on neuron signal
				x[l][j] = f.apply(s[l][j]);

			}
		}
			
		// Error of outputs in layer l
		double[] d;

		// Error of outputs in layer l-1
		double[] d_;

		// Compute error of each output in last layer (l = L)
		d = new double[target.length];
		for (int j = 0; j < d.length; j++)
			d[j] = 2 * (x[L - 1][j] - target[j]) * fD.apply(s[L-1][j]);
		
		// Back-propagation
		// For each layer l
		for (int l = L - 1; l > 0; l--) {

			// Initialize errors of layer l-1
			d_ = new double[x[l - 1].length];

			// For each input i
			for (int i = 0; i < x[l - 1].length; i++) {

				// Compute errors of layer l-1
				double sum = 0;
				for (int j = 0; j < x[l].length; j++) {
					sum += w[l][j][i] * d[j];
					w[l][j][i] -= learningRate * x[l-1][i] * d[j];
				}
				d_[i] = (1 - Math.pow(x[l - 1][i], 2)) * sum;

			}

			// If there are more layers
			if (l > 1)
				// Set error of layer accordingly
				d = d_;

		}

	}

}
