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

    public double[] train(double[] input, double[] ideal, double rate) {

        // Input values of layers + output of final layer: x[layer][]
        double[][] x = new double[L + 1][];
        x[0] = input;

        // Signal passed to neuron in a layer: s[layer][]
        double[][] s = new double[L][];

        // Partial derivative of E_total with respect to neuron output:
        double[] d = new double[w[L - 1].length];

        // Indication whether network output matches ideal value
        boolean pass = true;

        // Forward run
        // For each layer (l)
        for (int l = 0; l < L; l++) {

            // Number of neurons
            final int J = w[l].length;

            // Initialize neuron signal
            s[l] = new double[J];

            // Initialize neuron output (i.e. input of next layer)
            x[l + 1] = new double[J];

            // For each neuron (j)
            for (int j = 0; j < J; j++) {

                // Compute neuron signal
                s[l][j] = recallSignal(l, j, x[l]);

                // Compute output of neuron (i.e. input of next layer):
                // Apply activation function on neuron signal
                x[l + 1][j] = f.apply(s[l][j]);

                // If last layer, compare output with target (compute error)
                if (l == L - 1) {

                    // Compute neuron output error
                    d[j] = (x[l + 1][j] - ideal[j]);
                    if (x[l + 1][j] != ideal[j]) {
                        pass = false;
                    }

                }

            }

        }

        // If network output did not match ideal value
        if (!pass) {

            // Number of inputs (includig bias) in layer
            int I;

            // Number of outputs in layer
            int J;

            // Error with regards to weight of previous layer (l-1)
            double[] d_ = d;

            // Back-propagation
            // For each layer l
            for (int l = L - 1; l >= 0; l--) {

                // Retreive error with regards to weight
                d = d_;

                // If not first layer: use output from previous layer as input
                if (l > 0) {
                    d_ = new double[w[l - 1].length];
                }

                // Retreive number of inputs (including bias) in layer
                I = w[l][0].length;

                // Retreive number of outputs in layer
                J = w[l].length;

                // For each neuron j in layer l
                for (int j = 0; j < J; j++) {

                    double dO_dS = fD.apply(s[l][j]);

                    // For each input i in l (except bias input)
                    for (int i = 0; i < I - 1; i++) {

                        // Compute error with regards to weight:
                        // error = d * dO_dS * d2, where
                        // d:  Partial derivative of E_total with respect to neuron output
                        // dO_dS: P. d. of neuron output with regards to neuron signal
                        // d2: P. d. of neuron signal with regards to weight (i.e. neuron output)
                        double error = d[j] * dO_dS * x[l][i];

                        // If not first layer
                        if (l > 0) {

                            // Sum partial derivative of E_total with respect to
                            // neuron outputs
                            d_[i] += d[j] * dO_dS * w[l][j][i];

                        }

                        // Update weight
                        w[l][j][i] -= rate * error;

                    }

                    // Update bias weight
                    if (trainBias) {
                        double biasError = d[j] * dO_dS * biasInput;
                        w[l][j][I - 1] -= rate * biasError;
                    }

                }

            }

        }

        return x[L - 1];

    }

}
