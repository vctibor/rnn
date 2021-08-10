use rand::prelude::*;
use rand_distr::StandardNormal;
use Iterator;
use nalgebra::{Const, DMatrix, DVector, Dim, Dynamic};

// I rename standard Rust container from Vec to List so it isn't confused with Vector from linear algebra.
type List<T> = Vec<T>;

/// Bias is one value per node.
/// So we have Vec for each layer and in each Vec we have f64 bias for each node in this layer.
/// (First layer is input layer, therefore doesn't have biases or weights.)
type Biases = List<DVector<f64>>;

/// Each node has weight for each input, that means for every node in previous layer.
/// Therefore we have Vec for each layer and in each layer we have Vec for every node,
/// containing weight per every node in previous layer.
type Weights = List<DMatrix<f64>>;

/// Vector of (x, y) tuples, where `x` is input Vector, which has 
/// to be the same size as the first layer of Neural Net, and `y`
/// is expected result in form of index of node in last layer 
/// with highest activation value.
type TrainingData = List<(DVector<f64>, usize)>;

/// The sigmoid function.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/*
/// Derivative of the sigmoid function.
fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0-sigmoid(z))
}
*/

pub struct Network {
    pub num_layers: usize,
    pub sizes: Vec<usize>,
    pub biases: List<DVector<f64>>,
    pub weights: List<DMatrix<f64>>
}

impl Network {
    
    pub fn new(sizes: Vec<usize>) -> Network {

        let num_layers = sizes.len();
        let mut rng = rand::thread_rng();

        // Bias is one value per node.
        // So we have Vec for each layer and in each Vec we have f64 bias for each node in this layer.
        // (First layer is input layer, therefore doesn't have biases or weights.)
        let biases: List<DVector<f64>> = sizes
            .iter()
            .map(|layer_size| DVector::<f64>::from_iterator_generic(
                Dynamic::new(*layer_size), 
                Const::from_usize(1), 
                (&mut rng).sample_iter(StandardNormal)
            ))
            .collect();

        // Each node has weight for each input, that means for every node in previous layer.
        // Therefore we have Vec for each layer and in each layer we have Vec for every node,
        // containing weight per every node in previous layer.
        let weights: List<DMatrix<f64>> = sizes[..sizes.len()].iter()
            .zip(&sizes[1..])
            .map(|(previous_layer, this_layer)| DMatrix::<f64>::from_iterator_generic(
                Dynamic::new(*previous_layer),
                Dynamic::new(*this_layer),
                (&mut rng).sample_iter(StandardNormal)
            ))
            .collect();

        Network {
            num_layers,
            sizes,
            biases,
            weights
        }
    }

    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    fn evaluate(&self, test_data: &TrainingData) -> usize {
        let results: Vec<(usize, usize)> = test_data.iter()
            .map(|(x, y)| (self.feedforward(x.clone()).argmax().0, *y) )
            .collect();
        let correct_results: usize = results.iter().filter(|(x, y)| x == y).count();
        correct_results
    }

    /// Train the neural network using mini-batch stochastic
    /// gradient descent.  The "training_data" is a list of tuples
    /// "(x, y)" representing the training inputs and the desired
    /// outputs.  The other non-optional parameters are
    /// self-explanatory.  If "test_data" is provided then the
    /// network will be evaluated against the test data after each
    /// epoch, and partial progress printed out.  This is useful for
    /// tracking progress, but slows things down substantially.
    ///
    /// It is called *stochastic* gradient descent, because we select
    /// random sample from entire training set.
    ///
    /// This function is called `stochastic_gradient_descent` in the book.
    pub fn train(&mut self,
        mut training_data: TrainingData,
        epochs: i32,
        mini_batch_size: usize,
        learning_rate: f64,                 // eta is learning rate η.
        test_data: Option<TrainingData>)
    {
        let mut rng = rand::thread_rng();

        for epoch in 0..epochs {

            training_data.shuffle(&mut rng);

            let mini_batches: Vec<TrainingData> = (0..training_data.len()).step_by(mini_batch_size).map(|k|
                training_data[k..k+mini_batch_size].try_into().unwrap()
            ).collect();

            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, learning_rate);
            }

            if let Some(test_data) = &test_data {
                println!("Epoch {}: {}/{}", epoch, self.evaluate(test_data), test_data.len());
            } else {
                println!("Epoch {} complete.", epoch)
            }
        }
    }

    /// Get network's output for given input.
    fn feedforward(&self, mut input: DVector<f64>) -> DVector<f64> {
        assert_eq!(input.len(), self.sizes[0]); // Input has to be vector of size equal to number of input nodes.
        for (biases, weights) in self.biases.iter().zip(&self.weights) {
            input = ((weights * input) + biases).map(sigmoid);
        }
        input
    }

    /// Helper function to get zeroed weight and biases matrices.
    fn nabla(&self) -> (Biases, Weights) {

        let nabla_biases: List<DVector<f64>> = self.sizes
            .iter()
            .map(|layer_size| DVector::<f64>::zeros(*layer_size))
            .collect();

        let nabla_weights: List<DMatrix<f64>> = self.sizes[..self.sizes.len()].iter()
            .zip(&self.sizes[1..])
            .map(|(previous_layer, this_layer)|
                DMatrix::<f64>::zeros(*previous_layer, *this_layer)
            )
            .collect();

        (nabla_biases, nabla_weights)
    }

    /// Return a tuple ``(nabla_b, nabla_w)`` representing the
    /// gradient for the cost function C_x.  ``nabla_b`` and
    /// ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    /// to ``self.biases`` and ``self.weights``.
    fn backpropagate(&self, input_layer: &Vec<f64>, label: &usize) -> (Biases, Weights) {
        /*
        // `input_layer` is `x` in the book
        // `label` is `y` in the book

        /*
            x is training input, 28x28 -> 784 dimensional vector
            y is desired output, 10-dimensional vector


            The notation ‖v‖ just denotes the usual length function for a vector v.
            We'll call C the quadratic cost function; it's also sometimes known as the mean squared error or just MSE. 
        */

        // We have to keep track where in Python implementation uses vectorized result and where label.
        let desired_activations = label_to_activations(*label, *self.sizes.last().unwrap());

        let (mut nabla_biases, mut nabla_weights) = self.nabla();

        // feedforward

        // list to store all the activations, layer by layer
        let mut activations = vec![input_layer.clone()];

        // list to store all the z vectors, layer by layer
        let mut zs = vec![];

        for (b, w) in self.biases.iter().zip(&self.weights) {
            let z = add_vectors((&dot_product(w, activations.last().unwrap()), b));

            let activation = z.iter().map(|a| sigmoid(*a)).collect();

            zs.push(z);
            
            activations.push(activation);
        }

        // backward pass
        let delta = {
            let output_activations = activations.last().unwrap();
            let sigmoid_prime_z: Vec<f64> = zs.last().unwrap().iter().map(|a|sigmoid_prime(*a)).collect();
            let cost_derivative = self.cost_derivative(output_activations, &desired_activations);
            multiply_vectors(&cost_derivative, &sigmoid_prime_z)
        };

        // delta is calculated correctly
        
        let n = nabla_biases.len() - 1;
        nabla_biases[n] = delta.clone();

        let n = nabla_weights.len() - 1;
        let nn = activations.len() - 2;

        nabla_weights[n] = multiply_vectors4(&delta, &activations[nn]);

        // nabla_w[-1] is correct
        // nabla_b[-1] is correct
        
        // Note that the variable l in the loop below is used a little
        // differently to the notation in Chapter 2 of the book.  Here,
        // l = 1 means the last layer of neurons, l = 2 is the
        // second-last layer, and so on.  It's a renumbering of the
        // scheme in the book, used here to take advantage of the fact
        // that Python can use negative indices in lists.
        for l in 2..self.num_layers as i32 {
            
            let z = {
                let n = (zs.len() as i32 - l) as usize;
                zs[n].clone()
            };
            
            let delta = {
                let sigmoid_prime_z = z.iter().map(|a| sigmoid_prime(*a)).collect();

                let w = get(&self.weights, -l+1); // w is correct

                let product = multiply_vectors3(w, &delta);

                multiply_vectors(&product, &sigmoid_prime_z)
            };

            let n = (activations.len() as i32 - l -1) as usize;

            let weight = multiply_vectors4(&delta, &activations[n]);
            
            put(&mut nabla_biases, delta, -l);
            put(&mut nabla_weights, weight, -l);
        }

        (nabla_biases, nabla_weights)
        */

        panic!();
    }

    fn cost_derivative(&self, actual_activations: &Vec<f64>, expected_activations: &Vec<f64>) -> Vec<f64> {
        actual_activations.into_iter().zip(expected_activations)
            .map(|(x, y)| x-y)
            .collect()
    }

    /// Update the network's weights and biases by applying
    /// gradient descent using backpropagation to a single mini batch.
    fn update_mini_batch(&mut self, mini_batch: TrainingData, learning_rate: f64) {

        /*
        let (mut nabla_biases, mut nabla_weights) = self.nabla();

        for (x, y) in &mini_batch {

            let (delta_nabla_biases, delta_nabla_weights) = self.backpropagate(x, y);
    
            nabla_biases = nabla_biases.iter().zip(&delta_nabla_biases)
                .map(add_vectors)
                .collect();

            nabla_weights = nabla_weights.iter().zip(&delta_nabla_weights)
                .map(|(nw, dnw )| {
                    nw.iter().zip(dnw).map(add_vectors).collect()
                })
                .collect();
        }

        let rate = learning_rate / (mini_batch.len() as f64);

        let new_weights: Weights = self.weights.iter().zip(&nabla_weights)
            .map(|(w, nw)| subtract_matrices(w, &multiply_matrix_by_scalar(nw, rate)))
            .collect();

        let new_biases: Biases = self.biases.iter().zip(&nabla_biases)
            .map(|(b, nb)| subtract_vectors(b, &multiply_vector_by_scalar(nb, rate)))
            .collect();
        
        self.weights = new_weights;
        self.biases = new_biases;
        */

        panic!();
    }
}