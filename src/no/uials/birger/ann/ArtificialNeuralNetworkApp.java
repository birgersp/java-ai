package no.uials.birger.ann;

public class ArtificialNeuralNetworkApp {

	private void orTest() {

		Neuron neuron = new Neuron(2);
		double rate = 0.25;
		int workouts = 0;
		int maxWorkouts = 1000;
		int attempts = 1000000;

		double[] x = new double[2];
		double expectation;
		double result;

		boolean pass = false;
		while (!pass && workouts <= maxWorkouts) {

			pass = true;
			for (int i = 0; i < attempts; i++) {

				x[0] = Math.random() >= 0.5 ? 1 : 0;
				x[1] = Math.random() >= 0.5 ? 1 : 0;
				result = (neuron.recall(x) >= 0.9 ? 1 : 0);
				expectation = x[0] + x[1] >= 1 ? 1 : 0;

				if (result != expectation) {

					pass = false;
					neuron.train(x, expectation, rate);
					workouts++;
					break;

				}

			}

		}

		if (pass)
			System.out.println("It took " + workouts + " workouts to handle " + attempts + " attempts");

	}

	public ArtificialNeuralNetworkApp() {

		orTest();

	}

	public static void main(String[] args) {

		new ArtificialNeuralNetworkApp();

	}

}
