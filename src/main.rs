// Disable unused warning while in dev, remove later.
#![allow(dead_code)]


//#![feature(const_generics)]
//#![feature(const_evaluatable_checked)]
//#![feature(iter_zip)]

use std::{convert::TryInto};
use rand::prelude::*;

use Iterator;

mod mnist_loader;

/// Vector of (x, y) tuples, where `x` is input Vector, which has 
/// to be the same size as the first layer of Neural Net, and `y`
/// is expected result in form of index of node in last layer 
/// with highest activation value.
type TrainingData = Vec<(Vec<f64>, usize)>;

/// The sigmoid function.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Derivative of the sigmoid function.
fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0-sigmoid(z))
}

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


/* ---- */

fn add_vectors(input: (Vec<f64>, Vec<f64>)) -> Vec<f64> {
    input.0.into_iter().zip(input.1)
        .map(|(a,b)| a+b).collect()
}

fn multiply_vector_by_scalar(a: &Vec<f64>, b: f64) -> Vec<f64> {
    let mut collector = Vec::with_capacity(a.len());
    for x in a {
        collector.push(x * b);
    }
    collector
}

fn multiply_matrix_by_scalar(a: &Vec<Vec<f64>>, b: f64) -> Vec<Vec<f64>> {
    let mut collector = Vec::with_capacity(a.len());
    for aa in a {
        let mut collectora = Vec::with_capacity(aa.len());
        for aaa in aa {
            collectora.push(aaa * b);
        }
        collector.push(collectora);
    }
    collector
}

fn subtract_vectors(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut collector = Vec::with_capacity(a.len());
    for aa in a {
        for bb in b {
            collector.push(aa - bb);
        }
    }
    collector
}

fn subtract_matrices(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut collector = Vec::with_capacity(a.len());
    for aa in a {
        for bb in b {
            let mut collectora = Vec::with_capacity(aa.len());
            for aaa in aa {
                for bbb in bb {
                    collectora.push(aaa - bbb);
                }
            }
            collector.push(collectora);
        }
    }
    collector
}

/// Cross product
/// https://nrich.maths.org/2393
fn multiply_vectors(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.into_iter().zip(b).map(|(a, b)|a*b).collect()
}

/* ---- */


// https://gist.github.com/imbolc/dd0b439a7106ad621eaa1cf4df4a4152
fn dot_product(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let mut prod: Vec<f64> = Vec::with_capacity(vector.len());
    for row in matrix {
        let mut cell = 0f64;
        for (a, b) in row.iter().zip(vector.iter()) {
            cell += a * b;
        }
        prod.push(cell);
    }
    prod
}

/*
fn multiply_vectors2(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut res = vec![];
    for aa in a {
        for bb in b {
            res.push(aa*bb);
        }
    }
    res
}
*/

fn multiply_vectors3(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let mut res = vec![];
    for aa in a {
        for aaa in aa {
            for bb in b {
                res.push(aaa*bb);
            }
        }
    }
    res
}

fn multiply_vectors4(a: &Vec<f64>, b: &Vec<f64>) -> Vec<Vec<f64>> {
    let mut res = vec![];
    for aa in a {
        let mut inner = vec![];
        for bb in b {
            inner.push(aa * bb);
        }
        res.push(inner);
    }
    res
}



fn get<T>(vec: &Vec<T>, ix: i32) -> &T {
    let n = ((vec.len() as i32) + ix) as usize;
    &vec[n]
}

fn put<T>(vec: &mut Vec<T>, val: T, ix: i32) {
    let n = ((vec.len() as i32) + ix) as usize;
    vec[n] = val;
}


/// Bias is one value per node.
/// So we have Vec for each layer and in each Vec we have f64 bias for each node in this layer.
/// (First layer is input layer, therefore doesn't have biases or weights.)
type Biases = Vec<Vec<f64>>;

/// Each node has weight for each input, that means for every node in previous layer.
/// Therefore we have Vec for each layer and in each layer we have Vec for every node,
/// containing weight per every node in previous layer.
type Weights = Vec<Vec<Vec<f64>>>;

struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>
}

impl Network {
    
    fn new(sizes: Vec<usize>) -> Network {

        let num_layers = sizes.len();

        // NOTE:
        // Source material uses Gaussian distribution to 
        // initialize weights and biases, but we are using uniform distribution instead.
        // It shouldn't cause much issues, but it is starting point for investigation
        // in case the network doesn't behave the way we expect.
        let mut rng = rand::thread_rng();

        // Bias is one value per node.
        // So we have Vec for each layer and in each Vec we have f64 bias for each node in this layer.
        // (First layer is input layer, therefore doesn't have biases or weights.)
        let biases: Vec<Vec<f64>> = {
            let mut biases = Vec::new();
            for layer_size in &sizes[1..] {
                let v: Vec<f64> = (0..*layer_size).map(|_| rng.gen_range(-1f64..1f64)).collect();
                biases.push(v);
            }
            biases.try_into().unwrap()
        };

        // Each node has weight for each input, that means for every node in previous layer.
        // Therefore we have Vec for each layer and in each layer we have Vec for every node,
        // containing weight per every node in previous layer.
        let weights: Vec<Vec<Vec<f64>>> = {

            let mut weights = Vec::new();

            let a = &sizes[..sizes.len()];
            let b = &sizes[1..];

            let zipped: Vec<(&usize, &usize)> = a.into_iter().zip(b).collect();

            for (inputs, layer_size) in zipped {

                let mut node_weights = Vec::new();

                for _node in 0..*layer_size {
                    let v: Vec<f64> = (0..*inputs).map(|_| rng.gen_range(-1f64..1f64)).collect();
                    node_weights.push(v);
                }

                weights.push(node_weights);
            }

            weights.try_into().unwrap()
        };

        Network {
            num_layers,
            sizes,
            biases,
            weights
        }
    }

    /// Get network's output for given input.
    fn feedforward(&self, input: &Vec<f64>) -> Vec<f64> {

        // Input has to be vector of size equal to number of input nodes.
        assert_eq!(input.len(), self.sizes[0]);

        // Intermediate result between each layer.
        let mut a = input.clone();

        for (biases, weights) in self.biases.clone().into_iter().zip(self.weights.clone()) {
            let dot_product = dot_product(&weights, &a);

            assert_eq!(biases.len(), dot_product.len());

            let mut biases = biases.clone();
            biases.reverse();
            
            // dot product + biases
            let dot_product_plus_biases: Vec<f64> = dot_product.into_iter().map(|x| x + biases.pop().unwrap()).collect();

            a = dot_product_plus_biases.into_iter().map(sigmoid).collect();
        }

        a
    }


    /// Helper function to get zeroed weight and biases matrices.
    fn nabla(&self) -> (Biases, Weights) {
        let nabla_biases: Vec<Vec<f64>> = self.biases.clone().into_iter()
            .map(|b| vec![0f64; b.len()])
            .collect::<Vec<Vec<f64>>>()
            .try_into().unwrap();

        let nabla_weights: Vec<Vec<Vec<f64>>> = self.weights.clone().into_iter()
            .map(|w| {
                w.into_iter().map(|ww| vec![0f64; ww.len()]).collect::<Vec<Vec<f64>>>()
            }).collect::<Vec<Vec<Vec<f64>>>>()
            .try_into().unwrap();

        (nabla_biases, nabla_weights)
    }


    /// Return a tuple ``(nabla_b, nabla_w)`` representing the
    /// gradient for the cost function C_x.  ``nabla_b`` and
    /// ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    /// to ``self.biases`` and ``self.weights``.
    fn backpropagate(&self, input_layer: &Vec<f64>, label: &usize) -> (Biases, Weights) {
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

        //println!("\nbiases {:?}", nabla_biases);
        //println!("\nweights {:?}", nabla_weights);
        //println!("--------");

        // feedforward

        let mut activation = input_layer.clone();

        // list to store all the activations, layer by layer
        let mut activations = vec![activation.clone()];

        // list to store all the z vectors, layer by layer
        let mut zs = vec![];

        for (b, w) in self.biases.clone().into_iter().zip(self.weights.clone()) {
            let z = add_vectors((dot_product(&w, &activation), b.clone()));

            // z is calculated correctly

            zs.push(z.clone());
            activation = z.into_iter().map(sigmoid).collect();

            // activation is calculated correctly

            activations.push(activation.clone());
        }

        // backward pass
        let delta = {
            let output_activations: Vec<f64> = activations.last().unwrap().clone();
            let sigmoid_prime_z: Vec<f64> = zs.last().unwrap().clone().into_iter().map(sigmoid_prime).collect();
            let cost_derivative = self.cost_derivative(&output_activations, &desired_activations);
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
        
        //println!("\nbiases {:?}", nabla_biases);
        //println!("\nweights {:?}", nabla_weights);
        //println!("--------");

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
                let sigmoid_prime_z: Vec<f64> = z.into_iter().map(sigmoid_prime).collect();

                let w = get(&self.weights, -l+1); // w is correct

                //println!("w: {:?}", w);
                //println!("delta: {:?}", delta);

                let product = multiply_vectors3(w, &delta);

                //println!("product: {:?}", product);

                multiply_vectors(&product, &sigmoid_prime_z)
            };

            

            put(&mut nabla_biases, delta.clone(), -l);

            let n = (activations.len() as i32 - l -1) as usize;

            println!("delta {:?}", delta);
            println!("activation {:?}", activations[n]);

            let weight = multiply_vectors4(&delta, &activations[n]);
            
            put(&mut nabla_weights, weight, -l);

            println!("\nbiases {:?}", nabla_biases);
            println!("\nweights {:?}", nabla_weights);
            println!("--------");
        }

        /*
        expected output

        weights
        [
            array([
                [-0.00052112, -0.00097119, -0.00071062],
                [-0.00074712, -0.00139235, -0.00101879]
            ]),
            array([[-0.01376709, -0.01337866]])
        ]

        biases
        [
            array([[-0.00236875], [-0.00339598]]), array([[-0.02099017]])
        ]
        */


        (nabla_biases, nabla_weights)
    }

    fn cost_derivative(&self, actual_activations: &Vec<f64>, expected_activations: &Vec<f64>) -> Vec<f64> {
        actual_activations.into_iter().zip(expected_activations)
            .map(|(x, y)| x-y)
            .collect()
    }

    /// Update the network's weights and biases by applying
    /// gradient descent using backpropagation to a single mini batch.
    fn update_mini_batch(&mut self, mini_batch: TrainingData, learning_rate: f64) {

        let (mut nabla_biases, mut nabla_weights) = self.nabla();

        for (x, y) in &mini_batch {

            let (delta_nabla_biases, delta_nabla_weights) = self.backpropagate(x, y);

            nabla_biases = nabla_biases.into_iter().zip(delta_nabla_biases)
                .map(add_vectors)
                .collect::<Vec<Vec<f64>>>()
                .try_into().unwrap();

            nabla_weights = nabla_weights.into_iter().zip(delta_nabla_weights)
                .map(|(nw, dnw )| {
                    nw.into_iter().zip(dnw).map(add_vectors).collect()
                })
                .collect::<Vec<Vec<Vec<f64>>>>()
                .try_into().unwrap();
        }

        // What is learning_rate?
        let rate = learning_rate / (mini_batch.len() as f64);

        self.weights = self.weights.clone().into_iter().zip(nabla_weights)
            .map(|(w, nw)| subtract_matrices(&w, &multiply_matrix_by_scalar(&nw, rate)))
            .collect();

        self.biases = self.biases.clone().into_iter().zip(nabla_biases)
            .map(|(b, nb)| subtract_vectors(&b, &multiply_vector_by_scalar(&nb, rate)))
            .collect();
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
    fn stochastic_gradient_descent(&mut self,
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
                println!("Eoch {}: {}/{}", epoch, self.evaluate(test_data), test_data.len());
            } else {
                println!("Epoch {} complete.", epoch)
            }
        }
    }

    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    fn evaluate(&self, test_data: &TrainingData) -> usize {

        let results: Vec<(usize, usize)> = test_data.into_iter()
            .map(|(x, y)| (argmax(self.feedforward(x)), *y) )
            .collect();

        let correct_results: usize = results.into_iter().filter(|(x, y)| x == y).count();

        correct_results
    }


    fn print(&self) {
        println!("Layers: {}", self.num_layers);
        println!("Nodes per layer: {:?}", self.sizes);
        println!("Biases:");
        println!("{:#?}", self.biases);
        println!("Weights:");
        println!("{:#?}", self.weights);
    }
}


/// Takes label and turns it into vector of the same size as network output layer
/// (`activation_layer_size`) with zeroes in all positions except label-th one.
/// `label` has to be less than or equal to `activation_layer_size`.
fn label_to_activations(label: usize, activation_layer_size: usize) -> Vec<f64> {
    assert!(label <= activation_layer_size);
    let mut activation_layer = vec![0f64; activation_layer_size];
    activation_layer[label] = 1.0;
    activation_layer
}

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


fn main() {


    let (network, input) = sample_net();
    //net.print();

    let results = network.feedforward(&input);

    //println!("{:?}", results);

    let (b, w) = network.backpropagate(&input, &0);



    //println!("weights: {:?}", w);
    //println!("biases: {:?}", b);





    /*
    let train_data = mnist_loader::load_data("train");
    println!("{}", train_data.len());

    let test_data = mnist_loader::load_data("t10k");
    println!("{}", test_data.len());


    let a = train_data.last().unwrap();
    println!("{}", a.image.len());
    */
    
    /*
    let network = Network::new([3, 5, 2]);
    
    // network.print();

    let a = vec!(5.0, 4.0, 3.1);

    let result = network.feedforward(&a);

    println!("{:?}", result);

    //network.stochastic_gradient_descent();
    */
}
