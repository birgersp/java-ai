package no.uials.birger.ann;

public class Layer {

	private final Neuron[] neurons;

	public Layer(int inputs, int outputs) {

		neurons = new Neuron[outputs];
		for (int i = 0; i < neurons.length; i++)
			neurons[i] = new Neuron(inputs);
		
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
