package no.uials.birger.ann;

public class Network {

	private final Layer[] layers;
	
	public Network(Layer[] layers) {

		this.layers = layers;
	
	}
	
	public double[] recall(double[] input) {

		double[] output = input;
		for (Layer layer : layers)
			output = layer.recall(output);
		
		return output;
		
	}
	
}
