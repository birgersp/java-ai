package no.uials.birger.ann;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class SigmoidTest {

	public static void main(String[] args) {

		XYSeries w0 = new XYSeries("1 / e^(1+(-beta * x))");

		for (double i = -1; i <= 1; i += 0.01) {

			double result = Math.tanh(i*10);
			w0.add(i, result);
		}

		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(w0);

		JFreeChart chart = ChartFactory.createXYLineChart(null, null, null,
				dataset, PlotOrientation.VERTICAL, true, true, false);
		ChartFrame frame = new ChartFrame("Sigmoid test", chart);
		frame.pack();
		frame.setVisible(true);

	}

}
