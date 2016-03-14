package no.uials.birger.ann;

import java.util.Random;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class Test {

	public static void main(String[] args) {
		
		XYSeriesCollection dataset = new XYSeriesCollection();
        XYSeries xy = new XYSeries("oi");
        XYSeries xy2 = new XYSeries("Hei");
        Random rn = new Random();

        for (int i = 0; i < 10; i++) {
            int r = rn.nextInt();
            xy.add((double) i, (double) r);
            xy2.add((double) i, (double) r - 500);
        }
        dataset.addSeries(xy);
        dataset.addSeries(xy2);

        JFreeChart chart = ChartFactory.createXYLineChart(
                null, null, null, dataset, PlotOrientation.VERTICAL, true, true, false);
        ChartFrame frame = new ChartFrame("First", chart);
        frame.pack();
        frame.setVisible(true);

	}
	
}
