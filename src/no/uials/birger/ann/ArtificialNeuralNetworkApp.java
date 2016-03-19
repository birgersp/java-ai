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

		trainXOrNetwork();

	}

	public void trainXOrNetwork() {

		DoubleFunction<Double> tanh = (double x) -> Math.tanh(x);
		DoubleFunction<Double> tanhDiff = (double x) -> 1 - Math.pow(tanh.apply(x), 2);

		Network network = Network.getRandom(tanh, tanhDiff, 2, 2, 1);

		int workouts = 0;
		int maxWorkouts = 50;
		int attempts = 1000000;

		double[] x = new double[2];
		double[] ideal = new double[1];
		double result;

		boolean pass = false;
		while (!pass && workouts <= maxWorkouts) {

			pass = true;
			for (int i = 0; i < attempts && pass; i++) {

				x[0] = Math.random() >= 0.5 ? 1 : -1;
				x[1] = Math.random() >= 0.5 ? 1 : -1;
				result = (network.recall(x)[0] > 0 ? 1 : -1);
				
				ideal[0] = x[0] + x[1] == 0 ? 1 : -1;

				if (result != ideal[0]) {

					pass = false;
					network.train(x, ideal, 0.25);

					workouts++;

				}

			}

		}

		if (pass)
			System.out.println("It took " + workouts + " workouts to handle " + attempts + " attempts");
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

		JFreeChart chart = ChartFactory.createScatterPlot(null, null, null, dataset);
		chart.getXYPlot().setRenderer(new XYSplineRenderer());
		chart.setAntiAlias(true);
		chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.BLUE);
		chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
		ChartFrame frame = new ChartFrame("2D Network", chart);
		frame.pack();
		frame.setVisible(true);

	}

}
