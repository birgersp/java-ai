package no.uials.birger.ann;

public class Network {

	private final Layer[] layers;
	
	public Network(Layer... layers) {

		this.layers = layers;
	
	}
	
	public double[] recall(double[] input) {

		double[] output = input;
		for (Layer layer : layers)
			output = layer.recall(output);
		
		return output;
		
	}
	
	public Layer[] getLayers() {
	
		return layers;
	
	}
	
//	public void train(double[] input, double[] expectation, double learningRate) {
//		
//		// Forward run
//		double[][] outputs = new double[layers.length][];
//		outputs[0] = layers[0].recall(input);
//		for (int i = 1; i < layers.length; i++)
//			outputs[i] = layers[i].recall(outputs[i-1]);
//
//		// Backwards run
//		double error = 0;
//		for (int i = 0; i )
//		
//	}
	
}
