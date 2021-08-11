// Disable unused warning while in dev, remove later.
#![allow(dead_code)]

mod network;
mod mnist_loader;

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
