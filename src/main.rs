// Disable unused warning while in dev, remove later.
#![allow(dead_code)]


use Iterator;

mod network;
mod mnist_loader;



/// Returns index of largest element in vector,
/// assuming no NaNs or Infs.
fn argmax(a: Vec<f64>) -> usize {
    let mut largest_ix = 0;
    let mut largest_val = 0.0;
    for (ix, val) in a.into_iter().enumerate() {
        if val > largest_val {
            largest_ix = ix;
            largest_val = val;
        }
    }
    largest_ix
}

/*
fn list_to_vector(b: Vec<f64>) -> DVector<f64> {
    Vector::from_vec_generic(Dynamic::new(b.len()), ONE, b)
}
*/





/// Takes label and turns it into vector of the same size as network output layer
/// (`activation_layer_size`) with zeroes in all positions except label-th one.
/// `label` has to be less than or equal to `activation_layer_size`.
fn label_to_activations(label: usize, activation_layer_size: usize) -> Vec<f64> {
    assert!(label <= activation_layer_size);
    let mut activation_layer = vec![0f64; activation_layer_size];
    activation_layer[label] = 1.0;
    activation_layer
}

/*
/// Returns predefined random small network for testing purposes and input for it.
fn sample_net() -> (Network, Vec<f64>) {
    (Network {
        num_layers: 3,
        sizes: vec![3, 2, 1],
        biases: vec![
            vec![0.5, 0.4],             // biases of second layer
            vec![0.9]                   // biases of third layer
        ],
        weights: vec![
            vec![
                vec![0.2, 0.1, 0.2],    // weights for first neuron of second layer
                vec![0.1, 0.2, 0.2],    // weights for second neuron of second layer
            ],
            vec![
                vec![0.5, 0.7]
            ]
        ]
    },
    vec![0.22, 0.41, 0.3])
}
*/


fn main() {

    /*
    let mut net = Network::new(vec![784, 30, 10]);

    let (training_data, validation_data, test_data) = mnist_loader::load_all();

    let training_data = training_data.into_iter().map(|image| (image.image, image.classification as usize)).collect();
    let test_data = test_data.into_iter().map(|image| (image.image, image.classification as usize)).collect();

    // Book default values
    let epochs = 30;
    let mini_batch_size = 10;
    let learning_rate = 3.0;

    let epochs = 60;
    let mini_batch_size = 100;
    let learning_rate = 0.1;

    
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, Some(test_data));
    */
}
