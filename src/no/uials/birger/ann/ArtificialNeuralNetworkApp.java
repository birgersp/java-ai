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

	public static void main(String[] args) {

		new ArtificialNeuralNetworkApp();

	}

	private static void printDoubleArray(double[] array) {

		System.out.print("[ " + array[0]);

		for (int index = 1; index < array.length; index++)
			System.out.print(", " + array[index]);

		System.out.println(" ]");

	}

	public ArtificialNeuralNetworkApp() {

		// trainXOrNetwork();
		backpropagationExampleTest();

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

	public void trainXOrNetwork() {

		DoubleFunction<Double> f = (double x) -> 1 / (1 + Math.exp(-x));
		DoubleFunction<Double> fD = (double x) -> f.apply(x) * (1 - f.apply(x));

		double max = 1;
		double min = 0;
		double mid = (max - min) / 2;

		Network network = Network.getRandom(f, fD, 2, 1);

		int workouts = 0;
		int maxWorkouts = 50;
		int attempts = 1000000;

		double[] x = new double[2];
		double[] y;
		double[] ideal = new double[1];
		double result;

		boolean pass = false;
		while (!pass && workouts <= maxWorkouts) {

			pass = true;
			for (int i = 0; i < attempts && pass; i++) {

				x[0] = Math.random() >= 0.5 ? max : min;
				x[1] = Math.random() >= 0.5 ? max : min;
				y = network.recall(x);
				result = (y[0] >= mid ? max : min);

				ideal[0] = x[0] + x[1] > mid ? max : min;

				if (result != ideal[0]) {

					pass = false;
					network.train(x, ideal, 0.25);

					workouts++;

				}

			}

		}

		if (pass)
			System.out.println("It took " + workouts + " workouts to handle "
					+ attempts + " attempts");
		else
			System.err.println("Gave up training");

		show2DNetwork(network);

	}

	public void show2DNetwork(Network network) {

		XYSeries accepted = new XYSeries("Accepted");
		XYSeries rejected = new XYSeries("Rejected");
		double[] input = new double[2];
		double output;

		for (double y = -1; y <= 1; y += 0.025) {
			for (double x = -1; x <= 1; x += 0.025) {

				input[0] = x;
				input[1] = y;
				output = network.recall(input)[0];
				if (output > 0)
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

}
