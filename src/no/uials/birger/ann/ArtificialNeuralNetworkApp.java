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

	private void orTest(double rate) {

		XYSeries w0 = new XYSeries("w0");
		XYSeries w1 = new XYSeries("w1");
		XYSeries w2 = new XYSeries("w2");

		Neuron neuron = new Neuron(2);
		int workouts = 0;
		int maxWorkouts = 1000;
		int attempts = 1000000;

		double[] x = new double[2];
		double expectation;
		double result;

		boolean pass = false;
		while (!pass && workouts <= maxWorkouts) {

			pass = true;
			for (int i = 0; i < attempts && pass; i++) {

				x[0] = Math.random() >= 0.5 ? 1 : 0;
				x[1] = Math.random() >= 0.5 ? 1 : 0;
				result = (neuron.recall(x) > 0.9 ? 1 : 0);
				expectation = x[0] + x[1] >= 1 ? 1 : 0;

				if (result != expectation) {

					pass = false;
					neuron.train(x, expectation, rate);
					double[] w = neuron.getWeights();
					w0.add(workouts, w[0]);
					w1.add(workouts, w[1]);
					w2.add(workouts, w[2]);
					workouts++;

				}

			}

		}

		if (pass)
			System.out.println("It took " + workouts + " workouts to handle " + attempts + " attempts");

		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(w0);
		dataset.addSeries(w1);
		dataset.addSeries(w2);
		
		JFreeChart chart = ChartFactory.createXYLineChart(null, null, null, dataset, PlotOrientation.VERTICAL, true,
				true, false);
		ChartFrame frame = new ChartFrame("Learning rate: " + rate + ", pass: " + pass, chart);
		frame.pack();
		frame.setVisible(true);

		show2DNeuron(neuron);

	}

	public ArtificialNeuralNetworkApp() {

		orTest(0.25);

	}

	public void show2DNeuron(Neuron neuron) {

		XYSeries accepted = new XYSeries("Accepted");
		XYSeries rejected = new XYSeries("Rejected");
		double[] input = new double[2];
		double output;

		for (double y = 0; y <= 1.1; y += 0.05) {
			for (double x = 0; x <= 1.1; x += 0.05) {

				input[0] = x;
				input[1] = y;
				output = neuron.recall(input);
				if (output > 0.9)
					accepted.add(x, y);
				else
					rejected.add(x,y);

			}
		}

		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(accepted);
		dataset.addSeries(rejected);
		
		JFreeChart chart = ChartFactory.createScatterPlot(null, null, null, dataset);
		chart.getXYPlot().setRenderer(new XYSplineRenderer());
		chart.setAntiAlias(true);
		chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.GREEN);
		chart.getXYPlot().getRenderer().setSeriesPaint(1, Color.RED);
		ChartFrame frame = new ChartFrame("2D neuron", chart);
		frame.pack();
		frame.setVisible(true);

	}

	public static void main(String[] args) {

		new ArtificialNeuralNetworkApp();

	}

}
