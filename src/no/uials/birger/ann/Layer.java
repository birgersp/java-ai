package no.uials.birger.ann;

import java.util.function.DoubleFunction;

public class Layer {

	private final Neuron[] neurons;

	public static Layer getRandom(DoubleFunction<Double> f, int inputs, int outputs) {
		
		Neuron[] neurons = new Neuron[outputs];
		for (int i = 0; i < neurons.length; i++)
			neurons[i] = Neuron.getRandom(f, inputs);
		
		return new Layer(neurons);
		
	}
	
	public Layer(Neuron... neurons) {
		
		this.neurons = neurons;
		
	}

	public double[] recall(double[] input) {

		double[] output = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++)
			output[i] = neurons[i].recall(input);

		return output;

	}
	
	public Neuron[] getNeurons() {
	
		return neurons;
	
	}
	
}
