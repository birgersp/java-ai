package no.uials.birger.ann;

import java.awt.Color;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.function.DoubleFunction;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class ArtificialNeuralNetworkApp {

    public static int attempts = 0;

    public static void findFunction() {

        DoubleFunction<Double> f = (double x) -> Math.tanh(x);
        DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(f.apply(x), 2);

        final Network network = Network.getRandom(f, fD, 2, 2, 1);

        final XYSeries trainingSeries = new XYSeries("training");
        final XYSeries testingSeries = new XYSeries("testing");

//        XYSeries accepted = new XYSeries("Accepted");
//        XYSeries rejected = new XYSeries("Rejected");
        XYSeries neuralOutput = new XYSeries("Neural network");

        XYSeriesCollection errorData = new XYSeriesCollection();
        errorData.addSeries(trainingSeries);
        errorData.addSeries(testingSeries);

        JFreeChart errorChart = ChartFactory.createScatterPlot(null, null,
                null, errorData);
        errorChart.getXYPlot().setRenderer(new XYSplineRenderer());
        errorChart.setAntiAlias(true);
        errorChart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
        errorChart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
        ChartFrame errorFrame = new ChartFrame("Error rate", errorChart);

        int nTrain = 1000;
        int nTest = 500;
        final double[] trainingRate = {0.01};

        double range = 1;

        final DoubleFunction<Double> targetFunction = (double x) -> Math.sin((x - 0.5) * Math.PI);

        final double[][] trainingSet = new double[nTrain][];
        for (int i = 0; i < nTrain; i++) {
            trainingSet[i] = new double[2];
            trainingSet[i][0] = (Math.random() * 2 - 1) * range;
            trainingSet[i][1] = (Math.random() * 2 - 1) * range;
        }

        final double[][] testingSet = new double[nTest][];
        for (int i = 0; i < nTest; i++) {
            testingSet[i] = new double[2];
            testingSet[i][0] = (Math.random() * 2 - 1) * range;
            testingSet[i][1] = (Math.random() * 2 - 1) * range;
        }

        KeyListener keyListener = new KeyListener() {

            public void keyTyped(KeyEvent e) {
            }

            public void keyReleased(KeyEvent e) {
            }

            public void keyPressed(KeyEvent e) {

                switch (e.getKeyCode()) {
                    case KeyEvent.VK_1:
                        trainingRate[0] = 0.001;
                        break;
                    case KeyEvent.VK_2:
                        trainingRate[0] = 0.005;
                        break;
                    case KeyEvent.VK_3:
                        trainingRate[0] = 0.01;
                        break;
                    case KeyEvent.VK_4:
                        trainingRate[0] = 0.05;
                        break;
                    case KeyEvent.VK_5:
                        trainingRate[0] = 0.1;
                        break;
                    case KeyEvent.VK_ESCAPE:
                        System.exit(0);
                    case KeyEvent.VK_SPACE:
                        print3DDoubleArray(network.getWeights());
                        neuralOutput.clear();
                        for (double x = -range; x <= range; x += 0.05) {
                            for (double y = -range; y <= range; y += 0.05) {

                                double[] input2 = {x, y};
                                double[] output2 = network.recallAndActivate(input2);

                                if (output2[0] > 0.0) {
                                    neuralOutput.add(x, y);
                                    break;
                                }

                            }

                        }
                        break;
                    case KeyEvent.VK_ENTER:
                        int trainErrors = 0;
                        int testErrors = 0;
                        double[] input = null;
                        double[] output = null;
                        double[] target = new double[1];
                        for (int i = 0; i < nTrain; i++) {

                            input = trainingSet[i];
                            output = network.recallAndActivate(input);

                            if (input[1] < targetFunction.apply(input[0])) { // 0

                                target[0] = -1;
                                if (output[0] > 0) {
                                    trainErrors++;
                                }

                            } else { // 1

                                target[0] = 1;
                                if (output[0] <= 0) {
                                    trainErrors++;
                                }

                            }

                            network.train(input, target, trainingRate[0]);

                        }

                        for (int i = 0; i < nTest; i++) {

                            input = testingSet[i];
                            output = network.recallAndActivate(input);

                            if (input[1] < targetFunction.apply(input[0])) { // 0

                                target[0] = -1;
                                if (output[0] > 0) {
                                    testErrors++;
                                }

                            } else { // 1

                                target[0] = 1;
                                if (output[0] <= 0) {
                                    testErrors++;
                                }

                            }

                        }

                        double trainingError = (double) trainErrors / (double) nTrain;
                        double testingError = (double) testErrors / (double) nTest;
                        int epochs = trainingSeries.getItemCount();
                        trainingSeries.add(epochs, trainingError);
                        testingSeries.add(epochs, testingError);
                        break;
                    default:
                        break;
                }

            }
        };

        errorFrame.addKeyListener(keyListener);
        errorFrame.pack();
        errorFrame.setVisible(true);

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(neuralOutput);
//        dataset.addSeries(accepted);
//        dataset.addSeries(rejected);

        XYSeries sine = new XYSeries("Sine");
        double y;
        for (double x = -range; x <= range; x += 0.1) {
            y = targetFunction.apply(x);
            sine.add(x, y);
        }
        dataset.addSeries(sine);

        JFreeChart chart = ChartFactory.createScatterPlot(null, null, null,
                dataset);
        chart.getXYPlot().setRenderer(new XYSplineRenderer());
        chart.setAntiAlias(true);
        chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
        chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
        chart.getXYPlot().getRenderer().setSeriesPaint(2, Color.BLACK);
        ChartFrame frame = new ChartFrame("2D Network", chart);
        frame.addKeyListener(keyListener);
        frame.pack();
        frame.setVisible(true);

    }

    public static void interactiveNetwork() {

        DoubleFunction<Double> f = (double x) -> Math.tanh(x);
        DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(Math.tanh(x), 2);

        double[][][] w = {{{0.64, 0.53, 0.53}, {-0.37, 0.64, 0.59}}, {{-0.49, 0.31, -0.27}}};

        Network network = new Network(f, fD, w, -1, true);
        network.setBiasInput(-1);
        network.setTrainBias(false);

//        double[][][] w = network.getWeights();
        XYSeries[][][] wSeries = new XYSeries[w.length][][];

        System.out.print("[ ");
        for (int l = 0; l < w.length; l++) {

            System.out.print("[ ");
            wSeries[l] = new XYSeries[w[l].length][];
            for (int j = 0; j < w[l].length; j++) {

                System.out.print("[ ");
                wSeries[l][j] = new XYSeries[w[l][j].length];
                for (int i = 0; i < w[l][j].length; i++) {
                    System.out.print(w[l][j][i] + " ");
                    wSeries[l][j][i] = new XYSeries("w" + (i + 1));
                }
                System.out.print("] ");

            }
            System.out.print("] ");

        }
        System.out.println("]");

        XYSeries accepted = new XYSeries("Accepted");
        XYSeries rejected = new XYSeries("Rejected");

        KeyListener keyListener = new KeyListener() {

            public void keyTyped(KeyEvent e) {
            }

            public void keyReleased(KeyEvent e) {
            }

            public void keyPressed(KeyEvent e) {

                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {

                    System.exit(0);

                } else if (e.getKeyCode() == KeyEvent.VK_ENTER) {

                    boolean pass = true;
                    for (int a = -1; a < 2; a += 2) {

                        for (int b = -1; b < 2; b += 2) {

                            double[] x = {a, b};
                            double[] y = network.recallAndActivate(x);
                            double[] ideal = {(a + b == 0 ? 1 : -1)};
                            double result = y[0] > 0 ? 1 : -1;

                            if (result != ideal[0]) {

                                pass = false;
                                network.train(x, ideal, 0.01);

                            }

                        }

                    }

                    if (pass) {
                        System.err.println("Network training completed");
                        return;
                    }

                    accepted.clear();
                    rejected.clear();

                    for (double y = -1; y <= 1; y += 0.1) {
                        for (double x = -1; x <= 1; x += 0.1) {

                            double[] input = {x, y};
                            double[] output = network.recallAndActivate(input);

                            if (output[0] > 0.0) {
                                accepted.add(x, y);
                            } else {
                                rejected.add(x, y);
                            }

                        }
                    }

                    record(attempts++, network.getWeights(), wSeries);
                }

            }
        };

        for (int l = 0; l < wSeries.length; l++) {
            for (int j = 0; j < wSeries[l].length; j++) {

                XYSeriesCollection dataset = new XYSeriesCollection();

                for (int i = 0; i < wSeries[l][j].length; i++) {
                    dataset.addSeries(wSeries[l][j][i]);
                }

                JFreeChart chart = ChartFactory.createScatterPlot(null, null,
                        null, dataset);
                chart.getXYPlot().setRenderer(new XYSplineRenderer());
                chart.setAntiAlias(true);
                chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
                chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
                ChartFrame frame = new ChartFrame(
                        "Layer: " + (l + 1) + ", Neuron: " + (j + 1), chart);
                frame.addKeyListener(keyListener);
                frame.pack();
                frame.setVisible(true);

            }
        }

        System.out.println(w[1][0][2]);

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accepted);
        dataset.addSeries(rejected);

        record(attempts++, network.getWeights(), wSeries);

        JFreeChart chart = ChartFactory.createScatterPlot(null, null, null,
                dataset);
        chart.getXYPlot().setRenderer(new XYSplineRenderer());
        chart.setAntiAlias(true);
        chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
        chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
        ChartFrame frame = new ChartFrame("2D Network", chart);
        frame.addKeyListener(keyListener);
        frame.pack();
        frame.setVisible(true);

    }

    private static String arrayToString(double[] array) {

        StringBuilder sb = new StringBuilder();
        sb.append("[ " + array[0]);
        for (int i = 1; i < array.length; i++) {
            sb.append(", " + array[i]);
        }
        sb.append(" ]");
        return sb.toString();

    }

    public static void backpropagationExampleTest() {

        double[][][] w = {{{.15, .2, .35}, {.25, .3, .35}},
        {{.4, .45, .6}, {.5, .55, .6}}};
        double[] input = {.05, .1};
        double[] ideal = {.01, .99};

        DoubleFunction<Double> f = (double x) -> 1 / (1 + Math.exp(-x));
        DoubleFunction<Double> fD = (double x) -> f.apply(x) * (1 - f.apply(x));

        Network network = new Network(f, fD, w, 1, false);
        network.train(input, ideal, 0.5);

    }

    public static void main(String[] args) {

        new ArtificialNeuralNetworkApp();

    }

    private static void printDoubleArray(double[] array) {

        System.out.print("[ " + array[0]);

        for (int index = 1; index < array.length; index++) {
            System.out.print(", " + array[index]);
        }

        System.out.println(" ]");

    }

    public static void show2DNetwork(Network network) {

        XYSeries accepted = new XYSeries("Accepted");
        XYSeries rejected = new XYSeries("Rejected");

        for (double y = -1; y <= 1; y += 0.025) {
            for (double x = -1; x <= 1; x += 0.025) {

                double[] input = {x, y};
                double[] output = network.recallAndActivate(input);

                if (output[0] > 0.0) {
                    accepted.add(x, y);
                } else {
                    rejected.add(x, y);
                }

            }
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accepted);
        dataset.addSeries(rejected);

        JFreeChart chart = ChartFactory.createScatterPlot(null, null, null,
                dataset);
        chart.getXYPlot().setRenderer(new XYSplineRenderer());
        chart.setAntiAlias(true);
        chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
        chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
        ChartFrame frame = new ChartFrame("2D Network", chart);
        frame.pack();
        frame.setVisible(true);

    }

    public static void show1DNetwork(Network network) {

        XYSeries outputSeries = new XYSeries("Output");

        for (double x = -1; x <= 1; x += 0.025) {

            double[] input = {x};
            double[] output = network.recallAndActivate(input);
            outputSeries.add(x, output[0]);

        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(outputSeries);

        JFreeChart chart = ChartFactory.createScatterPlot(null, null, null,
                dataset);
        chart.getXYPlot().setRenderer(new XYSplineRenderer());
        chart.setAntiAlias(true);
        chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLACK);
        ChartFrame frame = new ChartFrame("1D Network", chart);
        frame.pack();
        frame.setVisible(true);

    }

    public static void record(double x, double[][][] w,
            XYSeries[][][] wSeries) {

        for (int l = 0; l < w.length; l++) {
            for (int j = 0; j < w[l].length; j++) {
                for (int i = 0; i < w[l][j].length; i++) {
                    wSeries[l][j][i].add(x, w[l][j][i]);
                }
            }
        }

    }

    public static void trainingAlgorithmExample() {

        DoubleFunction<Double> f = (double x) -> Math.tanh(x);
        DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(Math.tanh(x), 2);

        Network network = Network.getRandom(f, fD, 2, 2, 1);
        network.setBiasInput(-1);
        network.setTrainBias(false);

        double[][][] w = network.getWeights();
        XYSeries[][][] wSeries = new XYSeries[w.length][][];

        for (int l = 0; l < w.length; l++) {

            wSeries[l] = new XYSeries[w[l].length][];
            for (int j = 0; j < w[l].length; j++) {

                wSeries[l][j] = new XYSeries[w[l][j].length];
                for (int i = 0; i < w[l][j].length; i++) {
                    wSeries[l][j][i] = new XYSeries("w" + (i + 1));
                }

            }

        }

        System.out.println(w[1][0][2]);

        record(0, w, wSeries);

        int workouts = 0;
        int maxWorkouts = 500;

        boolean pass = false;
        while (!pass && workouts <= maxWorkouts) {

            pass = true;

            for (int a = -1; a < 2; a += 2) {

                for (int b = -1; b < 2; b += 2) {

                    double[] x = {a, b};
                    double[] y = network.recallAndActivate(x);
                    double[] ideal = {(a + b == 0 ? 1 : -1)};
                    double result = y[0] > 0 ? 1 : -1;

                    if (result != ideal[0]) {

                        pass = false;
                        network.train(x, ideal, 0.1);

                        workouts++;
                        record(workouts, w, wSeries);

                    }

                }

            }

        }

        record(workouts + 1, w, wSeries);

        if (pass) {
            System.out.println(
                    "It took " + workouts + " workouts to train an XOR");
        } else {
            System.err.println("Gave up training");
        }

        for (int l = 0; l < wSeries.length; l++) {
            for (int j = 0; j < wSeries[l].length; j++) {

                XYSeriesCollection dataset = new XYSeriesCollection();

                for (int i = 0; i < wSeries[l][j].length; i++) {
                    dataset.addSeries(wSeries[l][j][i]);
                }

                JFreeChart chart = ChartFactory.createScatterPlot(null, null,
                        null, dataset);
                chart.getXYPlot().setRenderer(new XYSplineRenderer());
                chart.setAntiAlias(true);
                chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
                chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
                ChartFrame frame = new ChartFrame(
                        "Layer: " + (l + 1) + ", Neuron: " + (j + 1), chart);
                frame.pack();
                frame.setVisible(true);

            }
        }

        System.out.println(w[1][0][2]);
        show2DNetwork(network);

    }

    public static void visualizeFunction(DoubleFunction<Double> f, double min,
            double max) {

        XYSeries w0 = new XYSeries("y");

        for (double i = min; i <= max; i += 0.01) {
            double result = f.apply(i);
            w0.add(i, result);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(w0);

        JFreeChart chart = ChartFactory.createXYLineChart(null, null, null,
                dataset, PlotOrientation.VERTICAL, true, true, false);
        ChartFrame frame = new ChartFrame("Function", chart);
        frame.pack();
        frame.setVisible(true);

    }

    public static double[][][] copy(double[][][] x) {

        int I1 = x.length;
        double[][][] y = new double[I1][][];
        for (int i1 = 0; i1 < I1; i1++) {

            int I2 = x[i1].length;
            y[i1] = new double[I2][];
            for (int i2 = 0; i2 < I2; i2++) {

                int I3 = x[i1][i2].length;
                y[i1][i2] = new double[I3];
                System.arraycopy(x[i1][i2], 0, y[i1][i2], 0, I3);

            }

        }

        return y;

    }

    public static void testANNSetups() {

        DoubleFunction<Double> f = (double x) -> Math.tanh(x);
        DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(Math.tanh(x), 2);
//
//        DoubleFunction<Double> f = (double x) -> 1 / (1 + Math.exp(-x));
//        DoubleFunction<Double> fD = (double x) -> f.apply(x) * (1 - f.apply(x));

        int successNetworks = 0;
        int maxEpochs = 10000;
        int N = 100;
        int totalEpochs = 0;
        int highEpochs = 0;
        int lowEpochs = maxEpochs;

        double[][][][] startingWeights = new double[N][][][];
        boolean[] failed = new boolean[N];

        double[] trainingRates = {0.05, 0.01};

        int n = 0;
        while (n < N) {

            Network network = Network.getRandom(f, fD, 2, 2, 1);
            network.setTrainBias(true);

            startingWeights[n] = copy(network.getWeights());

            int epochs = 0;
            boolean pass = false;
            double trainingRate;

            int rateIndex = 0;
            while (rateIndex < trainingRates.length && !pass) {

                trainingRate = trainingRates[rateIndex];

                epochs = 0;
                while (epochs < maxEpochs && !pass) {

                    pass = true;
                    for (int a = -1; a < 2; a += 2) {

                        for (int b = -1; b < 2; b += 2) {

                            double[] x = {a, b};
                            double[] y = network.recallAndActivate(x);
                            double[] ideal = {(a + b == 0 ? 1 : -1)};
                            double result = y[0] > 0 ? 1 : -1;

                            if (result != ideal[0]) {
                                pass = false;
                                network.train(x, ideal, trainingRate);
                            }

                        }

                    }

                    epochs++;

                }

                rateIndex++;

            }

            if (pass) {
                successNetworks++;
                totalEpochs += epochs;

                if (epochs > highEpochs) {
                    highEpochs = epochs;
                }

                if (epochs < lowEpochs) {
                    lowEpochs = epochs;
                }

                failed[n] = false;

            } else {

                failed[n] = true;

            }

            n++;

        }

        System.out.println("Failed starting weights:");
        for (int i = 0; i < N; i++) {
            if (failed[i]) {
                print3DDoubleArray(startingWeights[i]);
            }
        }

//        System.out.println();
//        System.out.println("Successful starting weights:");
//        for (int i = 0; i < N; i++) {
//            if (!failed[i]) {
//                print3DDoubleArray(startingWeights[i]);
//            }
//        }
        System.out.println();
        System.out.println(successNetworks + " of " + N + " successful");
        double averageAttempts = (double) totalEpochs / (double) successNetworks;
        System.out.println("Average epochs before successing: " + averageAttempts);
        System.out.println("Highest number of epochs before successing: " + highEpochs);
        System.out.println("Lowest number of epochs before successing: " + lowEpochs);

    }

    public static void print3DDoubleArray(double[][][] w) {

        System.out.print("[");
        int decimals = 2;

        int L = w.length;
        for (int l = 0; l < L; l++) {

            System.out.print("[");

            int J = w[l].length;
            for (int j = 0; j < J; j++) {

                System.out.print("[");

                int I = w[l][j].length;
                for (int i = 0; i < I; i++) {

                    double w0 = w[l][j][i];
                    System.out.print("\t" + (double) (Math.round(w0 * Math.pow(10, decimals)) / Math.pow(10, decimals)));

                    if (i != I - 1) {
                        System.out.print(",");
                    }

                }

                System.out.print("]");

                if (j != J - 1) {
                    System.out.print(",");
                }

            }

            System.out.print("]");

            if (l != L - 1) {
                System.out.print(",");
            }

        }

        System.out.println("]");

    }

    public static void customNetworkTest() {

        double[][][] w = {
            {{0.023128565894638936, 0.26438585799789094,
                0.40578767173669794},
            {-0.40215594432345825, 0.8512585283052834,
                0.40578767173669794}},
            {{-0.5421590355340142, 0.0112696760952758,
                0.20173599310633158}}};

        DoubleFunction<Double> f = (double x) -> Math.tanh(x);
        DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(Math.tanh(x), 2);

        Network network = new Network(f, fD, w, -1, true);
        show2DNetwork(network);

    }

    public ArtificialNeuralNetworkApp() {

        // trainingAlgorithmExample();
        // backpropagationExampleTest();
        // customNetworkTest();
//        interactiveNetwork();
        findFunction();
//        testANNSetups();
//
//        DoubleFunction<Double> f = (double x) -> 1 / (1 + Math.exp(-x));
//        DoubleFunction<Double> fD = (double x) -> f.apply(x) * (1 - f.apply(x));
//
//        double[][][] w = {{{0.15, 0.2, 0.35}, {0.25, 0.3, 0.35}}, {{0.4, 0.45, 0.6}, {0.5, 0.55, 0.6}}};
//
//        Network n = new Network(f, fD, w, 1, true);
//
//        double[] x = {0.05, 0.1};
//        double[] y = {0.01, 0.99};
//
//        n.train(x, y, 0.5);
//
//        System.out.println(n.recallAndActivate(x)[0]);

    }

}
