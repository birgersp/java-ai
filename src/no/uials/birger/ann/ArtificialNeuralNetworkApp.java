package no.uials.birger.ann;

import java.awt.Color;

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

	public ArtificialNeuralNetworkApp() {

		neuronTest(0.25);

	}

	private void neuronTest(double rate) {

		XYSeries w0 = new XYSeries("w0");
		XYSeries w1 = new XYSeries("w1");
		XYSeries w2 = new XYSeries("w2");

		Network network = new Network(new Layer(2, 1, 20));

		int workouts = 0;
		int maxWorkouts = 50;
		int attempts = 1000000;

		double[] w = network.getLayers()[0].getNeurons()[0].getWeights();
		w0.add(workouts, w[0]);
		w1.add(workouts, w[1]);
		w2.add(workouts, w[2]);

		double[] x = new double[2];
		double expectation;
		double result;

		boolean pass = false;
		while (!pass && workouts <= maxWorkouts) {

			pass = true;
			for (int i = 0; i < attempts && pass; i++) {

				x[0] = Math.random() >= 0.5 ? 1 : -1;
				x[1] = Math.random() >= 0.5 ? 1 : -1;
				result = (network.recall(x)[0] > 0.5 ? 1 : 0);
				expectation = x[0] + x[1] >= 0 ? 1 : 0;

				if (result != expectation) {

					pass = false;
					network.getLayers()[0].getNeurons()[0].train(x, expectation,
							rate);
					workouts++;

					w = network.getLayers()[0].getNeurons()[0].getWeights();
					w0.add(workouts, w[0]);
					w1.add(workouts, w[1]);
					w2.add(workouts, w[2]);

				}

			}

		}

		w = network.getLayers()[0].getNeurons()[0].getWeights();
		w0.add(workouts + 1, w[0]);
		w1.add(workouts + 1, w[1]);
		w2.add(workouts + 1, w[2]);

		if (pass)
			System.out.println("It took " + workouts + " workouts to handle "
					+ attempts + " attempts");

		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(w0);
		dataset.addSeries(w1);
		dataset.addSeries(w2);

		JFreeChart chart = ChartFactory.createXYLineChart(null, null, null,
				dataset, PlotOrientation.VERTICAL, true, true, false);
		ChartFrame frame = new ChartFrame(
				"Learning rate: " + rate + ", pass: " + pass, chart);
		frame.pack();
		frame.setVisible(true);

		show2DNetwork(network, "Trained network");

	}

	public void show2DNetwork(Network network, String title) {

		XYSeries accepted = new XYSeries("Accepted");
		XYSeries rejected = new XYSeries("Rejected");
		double[] input = new double[2];
		double output;

		for (double y = -1; y <= 1; y += 0.025) {
			for (double x = -1; x <= 1; x += 0.025) {

				input[0] = x;
				input[1] = y;
				output = network.recall(input)[0];
				if (output > 0.5)
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
		ChartFrame frame = new ChartFrame(title, chart);
		frame.pack();
		frame.setVisible(true);

	}

}
