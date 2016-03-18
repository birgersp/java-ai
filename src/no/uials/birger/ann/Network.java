package no.uials.birger.ann;

import java.util.function.DoubleFunction;

public class Network {

	public static Network getRandom(DoubleFunction<Double> f, int... layerOutputs) {

		// Number of layers
		int L = layerOutputs.length - 1;

		// Number of neurons for layer l
		int N;

		// Create layers
		Neuron[][] neurons = new Neuron[L][];

		// For each layer
		for (int l = 0; l < L; l++) {

			// Create neurons
			N = layerOutputs[l + 1];
			neurons[l] = new Neuron[N];

			// For each neuron
			for (int n = 0; n < N; n++)

				// Assign inputs
				neurons[l][n] = Neuron.getRandom(layerOutputs[l]);

		}

		return new Network(f, neurons);

	}

	private final DoubleFunction<Double> f;

	private final Neuron[][] neurons;

	public Network(DoubleFunction<Double> f, Neuron[]... neurons) {

		this.f = f;
		this.neurons = neurons;

	}

	public Neuron[][] getNeurons() {
		return neurons;
	}

	public double[] recall(double[] input) {

		// Inputs
		double[] x = input;

		// Outputs
		double[] y;

		// For each layer l
		for (int l = 0; l < neurons.length; l++) {

			y = recallLayer(l, x);
			x = y;

		}

		return x;

	}

	private double[] recallLayer(int l, double[] input) {

		// Layer output
		double[] y = new double[neurons[l].length];

		// For each neuron (output j)
		for (int j = 0; j < neurons[l].length; j++) {

			// Apply function on neuron signal
			y[j] = f.apply(recallSignal(l, j, input));

		}

		return y;

	}

	private double recallSignal(int l, int j, double[] input) {

		// Neuron weights
		double[] w = neurons[l][j].getWeights();

		// Compute neuron signal
		double s = w[0] * -1;

		// For each input i
		for (int i = 1; i < w.length; i++)
			s += w[i] * input[i - 1];

		return s;

	}

	public void train(double[] input, double[] target, double learningRate) {

		// Number of layers
		int L = neurons.length;

		// Output of layer outputs: x[layer][output]
		double[][] x = new double[L][];

		// Forward run
		// Compute output of each layer
		x[0] = input;
		for (int l = 1; l < L; l++)
			x[l] = recallLayer(l, x[l - 1]);

		// Error of outputs in layer l
		double[] d;

		// Error of outputs in layer l-1
		double[] d_;

		// Compute error of each output in last layer (l = L)
		d = new double[target.length];
		for (int j = 0; j < d.length; j++)
			d[j] = 2 * (x[L - 1][j] - target[j]) * (1 - Math.pow(x[L - 1][j], 2));
		
		// Back-propagation
		// For each layer l
		for (int l = L - 1; l > 0; l--) {

			// Initialize errors of layer l-1
			d_ = new double[x[l - 1].length];

			// For each input i
			for (int i = 0; i < x[l - 1].length; i++) {

				// Compute errors of layer l-1
				double sum = 0;
				for (int j = 0; j < x[l].length; j++)
					sum += neurons[l][j].getWeights()[i] * d[j];
				d_[i] = (1 - Math.pow(x[l - 1][i], 2)) * sum;

				// TODO: Update weights here

			}

			// If there are more layers
			if (l > 1)
				// Set error of layer accordingly
				d = d_;

		}

	}

}
