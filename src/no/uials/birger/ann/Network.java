package no.uials.birger.ann;

import java.util.Random;
import java.util.function.DoubleFunction;

public class Network {

	private final static double DEFAULT_BIAS_INPUT = -1;
	private final static boolean DEFAULT_TRAIN_BIAS = true;

	public static Network getRandom(DoubleFunction<Double> f,
			DoubleFunction<Double> fDerivative, int... layerOutputs) {

		int seed = 0;

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
				w[l][j] = getRandomNeuron(layerOutputs[l],
						System.currentTimeMillis() + seed++);

		}

		return new Network(f, fDerivative, w);

	}

	private static double[] getRandomNeuron(int inputs, long seed) {

		Random r = new Random(seed);

		double sqrtN = Math.sqrt(inputs);
		double[] w = new double[inputs + 1];
		for (int i = 0; i < w.length; i++)
			w[i] = 1 / sqrtN * (2 * r.nextDouble() - 1);

		return w;

	}

	private double biasInput = DEFAULT_BIAS_INPUT;
	private final DoubleFunction<Double> f;
	private final DoubleFunction<Double> fD;
	private final int L;
	private boolean trainBias = DEFAULT_TRAIN_BIAS;
	private final double[][][] w;

	public Network(DoubleFunction<Double> f, DoubleFunction<Double> fDerivative,
			double[][][] w) {

		this.f = f;
		this.fD = fDerivative;
		this.w = w;
		this.L = w.length;

	}

	public double[][][] getWeights() {

		return w;

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
		double s = 0;

		// For each input i
		for (int i = 0; i < w[l][j].length - 1; i++)
			s += w[l][j][i] * input[i];

		s += w[l][j][w[l][j].length - 1] * (double) biasInput;

		return s;

	}

	public void setBiasInput(double biasInput) {

		this.biasInput = biasInput;

	}

	public void setTrainBias(boolean trainBias) {

		this.trainBias = trainBias;

	}

	public double[] train(double[] input, double[] ideal, double rate) {

		// Output values of layer: x[layer][output index]
		double[][] x = new double[L][];

		// Signal passed to neuron in a layer: s[layer][output index]
		double[][] s = new double[L][];

		// Partial derivative of E_total with respect to neuron output:
		// d[neuron]
		double[] d = new double[w[L - 1].length];

		// Signals whether network output matches ideal value
		boolean pass = true;

		// "Bias error": sum of errors that each layer bias connects to
		double[] e = (trainBias ? new double[w.length] : null);

		// Forward run
		// Compute output of each layer
		for (int l = 0; l < L; l++) {

			// Number of neurons
			final int J = w[l].length;

			// Initialize signals and output
			x[l] = new double[J];
			s[l] = new double[J];

			// Initialize bias error
			if (trainBias)
				e[l] = 0;

			// For each neuron (output j)
			for (int j = 0; j < J; j++) {

				// Compute neuron signal
				if (l == 0)
					s[l][j] = recallSignal(l, j, input);
				else
					s[l][j] = recallSignal(l, j, x[l - 1]);

				// Apply function on neuron signal
				x[l][j] = f.apply(s[l][j]);

				// If last layer
				if (l == L - 1) {

					// Compute output error
					d[j] = -(ideal[j] - x[l][j]);
					if (x[l][j] != ideal[j])
						pass = false;

					// Compute bias error
					if (trainBias)
						e[l] += d[j];
					
				}

			}
		}

		// If network output did not match ideal value
		if (!pass) {

			// Partial derivative buffer
			double[] d_ = d;

			// Layer input
			double[] x_ = null;

			// Back-propagation
			// For each layer l
			for (int l = L - 1; l >= 0; l--) {

				// Use sum
				d = d_;

				// If not last layer: use output from previous layer as input
				if (l > 0) {
					x_ = x[l - 1];
					d_ = new double[w[l - 1].length];
				} else
					x_ = input;

				// For each neuron j in l
				for (int j = 0; j < w[l].length; j++) {

					// For each input i in l
					for (int i = 0; i < w[l][j].length - 1; i++) {

						// Compute error with regards to weight
						double d1 = fD.apply(s[l][j]);
						double d2 = x_[i];
						double error = d1 * d2 * d[j];

						// If not first layer
						if (l > 0) {

							// Sum partial derivative of E_total with respect to
							// neuron outputs
							d_[i] += d[j] * d1 * w[l][j][i];

							// Compute error of previous layer
							if (trainBias)
								e[l - 1] += d_[i];

						}

						// Update neuron weight
						w[l][j][i] -= rate * error;

					}

					// Update bias weight
					if (trainBias)
						w[l][j][w[l][j].length - 1] += e[l] * rate;

				}

			}

		}

		return x[L - 1];

	}

}
