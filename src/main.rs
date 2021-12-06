// Disable unused warning while in dev, remove later.
#![allow(dead_code)]

use nalgebra::DVector;

use crate::network::{Network, TrainingData};

mod network;
mod mnist_loader;

fn main() {

    let mut net = Network::new(vec![784, 30, 10]);

    let (training_data, validation_data, test_data) = mnist_loader::load_all();    

    let training_data: TrainingData = training_data.into_iter().map(|image|
        (DVector::<f64>::from_vec(image.image), image.classification as usize)
    ).collect();

    let test_data: TrainingData = test_data.into_iter().map(|image|
        (DVector::<f64>::from_vec(image.image), image.classification as usize)
    ).collect();
    

    // Book default values
    // epochs = 30;
    // mini_batch_size = 10;
    // learning_rate = 3.0;
    
    let epochs = 30;
    let mini_batch_size = 10;
    let learning_rate = 3.0;

    net.train(training_data, epochs, mini_batch_size, learning_rate, Some(test_data));
}
