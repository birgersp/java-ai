package no.uials.birger.ann;

import java.awt.Color;
import java.util.function.DoubleFunction;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class ArtificialNeuralNetworkApp {

	private static String arrayToString(double[] array) {

		StringBuilder sb = new StringBuilder();
		sb.append("[ " + array[0]);
		for (int i = 1; i < array.length; i++)
			sb.append(", " + array[i]);
		sb.append(" ]");
		return sb.toString();

	}

	public static void backpropagationExampleTest() {

		double[][][] w = { { { .15, .2, .35 }, { .25, .3, .35 } },
				{ { .4, .45, .6 }, { .5, .55, .6 } } };
		double[] input = { .05, .1 };
		double[] ideal = { .01, .99 };

		DoubleFunction<Double> f = (double x) -> 1 / (1 + Math.exp(-x));
		DoubleFunction<Double> fD = (double x) -> f.apply(x) * (1 - f.apply(x));

		Network network = new Network(f, fD, w);

		network.train(input, ideal, 0.5);

	}

	public static void main(String[] args) {

		new ArtificialNeuralNetworkApp();

	}

	private static void printDoubleArray(double[] array) {

		System.out.print("[ " + array[0]);

		for (int index = 1; index < array.length; index++)
			System.out.print(", " + array[index]);

		System.out.println(" ]");

	}

	public static void show2DNetwork(Network network) {

		XYSeries accepted = new XYSeries("Accepted");
		XYSeries rejected = new XYSeries("Rejected");

		for (double y = -1; y <= 1; y += 0.025) {
			for (double x = -1; x <= 1; x += 0.025) {

				double[] input = { x, y };
				double[] output = network.recall(input);

				if (output[0] > 0.0)
					accepted.add(x, y);
				else
					rejected.add(x, y);

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

	public static void record(double x, double[][][] w,
			XYSeries[][][] wSeries) {

		for (int l = 0; l < w.length; l++)
			for (int j = 0; j < w[l].length; j++)
				for (int i = 0; i < w[l][j].length; i++)
					wSeries[l][j][i].add(x, w[l][j][i]);

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
				for (int i = 0; i < w[l][j].length; i++)
					wSeries[l][j][i] = new XYSeries("w" + (i + 1));

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

					double[] x = { a, b };
					double[] y = network.recall(x);
					double[] ideal = { (a + b == 0 ? 1 : -1) };
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

		if (pass)
			System.out.println(
					"It took " + workouts + " workouts to train an XOR");
		else
			System.err.println("Gave up training");

		for (int l = 0; l < wSeries.length; l++)
			for (int j = 0; j < wSeries[l].length; j++) {

				XYSeriesCollection dataset = new XYSeriesCollection();

				for (int i = 0; i < wSeries[l][j].length; i++)
					dataset.addSeries(wSeries[l][j][i]);

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

	public static void customNetworkTest() {

		double[][][] w = {
				{ { 0.023128565894638936, 0.26438585799789094,
						0.40578767173669794 },
				{ -0.40215594432345825, 0.8512585283052834,
						0.40578767173669794 } },
				{ { -0.5421590355340142, 0.0112696760952758,
						0.20173599310633158 } } };

		DoubleFunction<Double> f = (double x) -> Math.tanh(x);
		DoubleFunction<Double> fD = (double x) -> 1 - Math.pow(Math.tanh(x), 2);

		Network network = new Network(f, fD, w);
		show2DNetwork(network);

	}

	public ArtificialNeuralNetworkApp() {

		trainingAlgorithmExample();
		// backpropagationExampleTest();
		// customNetworkTest();

	}

}
