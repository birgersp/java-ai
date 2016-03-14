package no.uials.birger.ann;

public class Layer {

	private final Neuron[] neurons;

	public Layer(int noOfInputs, int noOfNeurons) {

		neurons = new Neuron[noOfNeurons];
		for (Neuron n : neurons)
			n = new Neuron(noOfInputs);

	}

	public double[] recall(double[] input) {

		double[] output = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++)
			output[i] = neurons[i].recall(input);

		return output;

	}
	
}
