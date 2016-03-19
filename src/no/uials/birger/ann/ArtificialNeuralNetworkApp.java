package no.uials.birger.ann;

import java.awt.Color;
import java.util.function.DoubleFunction;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
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

		DoubleFunction<Double> f = (double x) -> 1 / (1 + Math.exp(-x));
		DoubleFunction<Double> fD = (double x) -> f.apply(x) * (1 - f.apply(x));

		double[] h1 = { .15, .2, .35 };
		double[] h2 = { .25, .3, .35 };

		double[] o1 = { .4, .45, .6 };
		double[] o2 = { .5, .55, .6 };

		double[][] l1 = { h1, h2 };
		double[][] l2 = { o1, o2 };

		double[][][] w = { l1, l2 };

		Network network = new Network(f, fD, w);

		double[] i = { .05, .1 };

		double[] ideal = {0.01, 0.99};

		network.train(i, ideal, 0.5);

	}

	public void show2DNetwork(Network network) {

		XYSeries accepted = new XYSeries("Accepted");
		XYSeries rejected = new XYSeries("Rejected");
		double[] input = new double[2];
		double output;

		for (double y = 0; y <= 1; y += 0.025) {
			for (double x = 0; x <= 1; x += 0.025) {

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
