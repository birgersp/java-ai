package no.birgersp.ann;

import java.awt.Color;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.function.DoubleFunction;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class ArtificialNeuralNetworkApp {

    private static void findFunction() {

        DoubleFunction<Double> f = (double x) -> Math.tanh(x);
        DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(f.apply(x), 2);

        final Network network = Network.getRandom(f, fD, 2, 20, 1);

        final XYSeries trainingSeries = new XYSeries("training");
        final XYSeries testingSeries = new XYSeries("testing");

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

        double range = 4;

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
        for (double x = -range; x <= range; x += 0.1) {
            sine.add(x, targetFunction.apply(x));
        }
        dataset.addSeries(sine);

        JFreeChart chart = ChartFactory.createScatterPlot(null, null, null,
                dataset);
        chart.getXYPlot().setRenderer(new XYSplineRenderer());
        chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
        chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
        chart.getXYPlot().getRenderer().setSeriesPaint(2, Color.BLACK);
        chart.getXYPlot().getRangeAxis().setAutoRange(false);
        ChartFrame frame = new ChartFrame("2D Network", chart);

        chart.setAntiAlias(true);

        frame.addKeyListener(keyListener);
        frame.pack();
        frame.setVisible(true);

        int epochs = 0;
        while (true) {

            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                e.printStackTrace();
                break;
            }

//            print3DDoubleArray(network.getWeights());
            neuralOutput.clear();

            for (double x = -range; x <= range; x += 0.1) {
                for (double y = -range; y <= range; y += 0.1) {

                    double[] input2 = {x, y};
                    double[] output2 = network.recallAndActivate(input2);

                    if (output2[0] > 0.0) {
                        neuralOutput.add(x, y);
                        break;
                    }

                }

            }

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

            if (epochs == 0 || epochs % 10 == 0) {
                double trainingError = (double) trainErrors / (double) nTrain;
                double testingError = (double) testErrors / (double) nTest;
                trainingSeries.add(epochs, trainingError);
                testingSeries.add(epochs, testingError);
            }

            epochs++;
        }

    }

    public static void main(String[] args) {

        ArtificialNeuralNetworkApp.findFunction();
        
    }

}
