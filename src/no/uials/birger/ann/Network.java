package no.uials.birger.ann;

import java.util.Random;
import java.util.function.DoubleFunction;

public class Network {

    public final static double DEFAULT_BIAS_INPUT = -1;
    public final static boolean DEFAULT_TRAIN_BIAS = true;

    public static Network getRandom(DoubleFunction<Double> f,
            DoubleFunction<Double> fDerivative, int... layerOutputs) {

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
            for (int j = 0; j < J; j++) {
                w[l][j] = getRandomNeuron(layerOutputs[l], f);
            }

        }

        return new Network(f, fDerivative, w, DEFAULT_BIAS_INPUT, DEFAULT_TRAIN_BIAS);

    }

    private static double[] getRandomNeuron(int inputs, DoubleFunction<Double> activation) {

        double min = activation.apply(Double.NEGATIVE_INFINITY);
        double max = activation.apply(Double.POSITIVE_INFINITY);
        double range = max - min;

        Random r = new Random();
        double sqrtN = Math.sqrt(inputs);
        double[] w = new double[inputs + 1];
        for (int i = 0; i <= inputs; i++) {
            w[i] = 1 / sqrtN * (range * r.nextDouble() + min);
        }

        return w;

    }

    private double biasInput;
    private final DoubleFunction<Double> f;
    private final DoubleFunction<Double> fD;
    private final int L;
    private boolean trainBias;
    private final double[][][] w;

    public Network(DoubleFunction<Double> f, DoubleFunction<Double> fDerivative,
            double[][][] w, double biasInput, boolean trainBias) {

        this.f = f;
        this.fD = fDerivative;
        this.w = w;
        this.L = w.length;
        this.biasInput = biasInput;
        this.trainBias = trainBias;

    }

    public double[][][] getWeights() {

        return w;

    }

    public double[] recallAndActivate(double[] input) {

        // Input values
        double[] x = input;

        // Neuron signal value (before activation)
        double s;

        // Output values
        double[] y = null;

        // Number of outputs
        int J;

        // For each layer l
        for (int l = 0; l < L; l++) {

            // Number of outputs of layer l
            J = w[l].length;

            // Layer output
            y = new double[J];

            // For each neuron (output j)
            for (int j = 0; j < J; j++) {

                // Compute neuron signal
                s = recallSignal(l, j, x);

                // Apply function on neuron signal
                y[j] = f.apply(s);

            }

            // Input of next layer is output of the current one
            x = y;

        }

        // Return the output
        return y;

    }

    private double recallSignal(int l, int j, double[] input) {

        // Compute output signal
        double s = 0;

        // Number of inputs in layer
        int I = w[l][j].length;

        // Neuron signal is sum of input values multiplied with their corresponding weights
        // For each input i
        for (int i = 0; i < I - 1; i++) {
            // Add weight times input value
            s += w[l][j][i] * input[i];
        }

        // Add bias weight times bias input (last input of layer)
        s += w[l][j][I - 1] * (double) biasInput;

        // Return output signal
        return s;

    }

    public void setBiasInput(double biasInput) {

        this.biasInput = biasInput;

    }

    public void setTrainBias(boolean trainBias) {

        this.trainBias = trainBias;

    }
    
    /**
     * Trains (by adjusting the weights of) the neural network.
     * 
     * @param input Input values
     * @param ideal Target output values
     * @param rate Training rate
     * @return The output of the network
     */
    public double[] train(double[] input, double[] ideal, double rate) {

        // "w" represents a three-dimensional array holding the input weights
        // "L" is stored as number of layers
        
        // Neuron signals
        double[][] s = new double[L][];
        // Neuron output
        double[][] x = new double[L][];
        // Partial derivative of weight error with regards to signal
        double[][] d = new double[L][];

        // For each layer
        for (int l = 0; l < L; l++) {

            // Number of neurons in layer
            final int J = w[l].length;

            s[l] = new double[J];
            x[l] = new double[J];
            d[l] = new double[J];

            // For each neuron
            for (int j = 0; j < J; j++) {

                // Compute neuron output
                s[l][j] = recallSignal(l, j, l == 0 ? input : x[l - 1]);
                x[l][j] = f.apply(s[l][j]);

                // If last layer, compute output error
                if (l == L - 1) {
                    d[l][j] = x[l][j] - ideal[j];
                }

            }

        }

        // For each layer (backwards)
        for (int l = L - 1; l >= 0; l--) {

            // Number of neurons in layer
            final int J = w[l].length;

            // For each layer
            for (int j = 0; j < J; j++) {

                // Number of inputs in layer
                final int I = w[l][j].length;

                // For each input
                for (int i = 0; i < I; i++) {

                    // If not first layer, compute error in l-1
                    if (l > 0 && i < I - 1) {
                        d[l - 1][i] += d[l][j] * w[l][j][i] * fD.apply(s[l - 1][i]);
                    }

                    // Update weight
                    w[l][j][i] -= rate * (i == I - 1
                            ? biasInput // Use bias input on bias weight
                            : (l == 0 ? input[i] : x[l - 1][i])) * d[l][j];

                }

            }

        }

        // Return output of last layer
        return x[L - 1];

    }

}
