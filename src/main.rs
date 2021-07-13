#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![feature(iter_zip)]

use std::{convert::TryInto, iter::zip};
use rand::prelude::*;

/// Vector of (x, y) tuples, where `x` is input Vector, which has 
///  to be the same size as the first layer of Neural Net, and `y`
///  is expected result in form of index of node in last layer 
///  with highest activation value.
type TrainingData = Vec<(Vec<f32>, usize)>;

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Returns index of largest element in vector,
///  assuming no NaNs or Infs.
fn argmax(a: Vec<f32>) -> usize {
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

fn add(input: (Vec<f32>, Vec<f32>)) -> Vec<f32> {
    zip(input.0, input.1).map(|(a,b)| a+b).collect::<Vec<f32>>()
}

fn mul(a: &Vec<f32>, b: f32) -> Vec<f32> {
    let mut collector = Vec::with_capacity(a.len());
    for x in a {
        collector.push(x * b);
    }
    collector
}

fn mull(a: &Vec<Vec<f32>>, b: f32) -> Vec<Vec<f32>> {
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

fn minus(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
    let mut collector = Vec::with_capacity(a.len());
    for aa in a {
        for bb in b {
            collector.push(aa - bb);
        }
    }
    collector
}

fn minuss(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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

/* ---- */


// https://gist.github.com/imbolc/dd0b439a7106ad621eaa1cf4df4a4152
fn dot_product(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
    let mut prod: Vec<f32> = Vec::with_capacity(vector.len());
    for row in matrix {
        let mut cell = 0f32;
        for (a, b) in row.iter().zip(vector.iter()) {
            cell += a * b;
        }
        prod.push(cell);
    }
    prod
}

struct Network<const N: usize> where [(); N-1]: {
    //num_layers: usize,
    sizes: [usize; N],
    biases: [Vec<f32>; N-1],
    weights: [Vec<Vec<f32>>; N-1]
}

impl<const N: usize> Network<N> where [(); N-1]: {
    
    fn new(sizes: [usize; N]) -> Network<N> where [(); N-1]: {

        let num_layers = N;

        /*
            NOTE:
            Source material uses Gaussian distribution to 
             initialize weights and biases, but we are using uniform distribution instead.
            It shouldn't cause much issues, but it is starting point for investigation
             in case the network doesn't behave the way we expect.
        */
        let mut rng = rand::thread_rng();

        /*
            Bias is one value per node.
            So we have Vec for each layer and in each Vec we have f32 bias for each node in this layer.

            [
                array([[-0.4651621 ], [ 0.8158959 ], [ 0.54096477]]),
                array([[-0.22998989]])
            ]

            (First layer is input layer, therefore doesn't have biases or weights.)
        */
        let biases: [Vec<f32>; N-1] = {
            let mut biases = Vec::new();
            for layer_size in &sizes[1..] {
                let v: Vec<f32> = (0..*layer_size).map(|_| rng.gen_range(-1f32..1f32)).collect();
                biases.push(v);
            }
            biases.try_into().unwrap()
        };

        /*
            Each node has weight for each input, that means for every node in previous layer.
            Therefore we have Vec for each layer and in each layer we have Vec for every node,
             containing weight per every node in previous layer.

            [
                array([
                    [-0.83729071, -0.73971336],
                    [ 0.70516901, -2.23510949],
                    [-1.57423188, -0.32834226]
                ]),
                array([
                    [ 0.12641327, -2.2249865 ,  2.14234274]
                ])
            ]
        */
        let weights: [Vec<Vec<f32>>; N-1] = {

            let mut weights = Vec::new();

            let a = &sizes[..sizes.len()];
            let b = &sizes[1..];

            let zipped: Vec<(&usize, &usize)> = zip(a, b).collect();

            for (inputs, layer_size) in zipped {

                let mut node_weights = Vec::new();

                for _node in 0..*layer_size {
                    let v: Vec<f32> = (0..*inputs).map(|_| rng.gen_range(-1f32..1f32)).collect();
                    node_weights.push(v);
                }

                weights.push(node_weights);
            }

            weights.try_into().unwrap()
        };

        Network::<N> {
            //num_layers,
            sizes,
            biases,
            weights
        }
    }

    fn feedforward(&self, input: &Vec<f32>) -> Vec<f32> {

        // Input has to be vector of size equal to number of input nodes.
        assert_eq!(input.len(), self.sizes[0]);

        // Intermediate result between each layer.
        let mut a = input.clone();

        for (biases, weights) in zip(&self.biases, &self.weights) {
            let dot_product = dot_product(weights, &a);

            assert_eq!(biases.len(), dot_product.len());

            let mut biases = biases.clone();
            biases.reverse();
            
            // dot product + biases
            let dot_product_plus_biases: Vec<f32> = dot_product.into_iter().map(|x| x + biases.pop().unwrap()).collect();

            a = dot_product_plus_biases.into_iter().map(sigmoid).collect();
        }

        a
    }

    /// Return a tuple ``(nabla_b, nabla_w)`` representing the
    /// gradient for the cost function C_x.  ``nabla_b`` and
    /// ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    /// to ``self.biases`` and ``self.weights``.
    fn backpropagate(&self, x: &Vec<f32>, y: &usize) -> ([Vec<f32>; N-1], [Vec<Vec<f32>>; N-1]) {
        panic!();
    }

    /// Update the network's weights and biases by applying
    /// gradient descent using backpropagation to a single mini batch.
    fn update_mini_batch(&mut self, mini_batch: TrainingData, learning_rate: f32) {

        let mut nabla_biases: [Vec<f32>; N-1] = self.biases.into_iter()
            .map(|b| vec![0f32; b.len()])
            .collect::<Vec<Vec<f32>>>()
            .try_into().unwrap();

        let mut nabla_weights: [Vec<Vec<f32>>; N-1] = self.weights.into_iter()
            .map(|w| {
                w.into_iter().map(|ww| vec![0f32; ww.len()]).collect::<Vec<Vec<f32>>>()
            }).collect::<Vec<Vec<Vec<f32>>>>()
            .try_into().unwrap();

        for (x, y) in &mini_batch {

            let (delta_nabla_biases, delta_nabla_weights) = self.backpropagate(x, y);

            nabla_biases = zip(nabla_biases, delta_nabla_biases)
                .map(add)
                .collect::<Vec<Vec<f32>>>()
                .try_into().unwrap();

            nabla_weights = zip(nabla_weights, delta_nabla_weights)
                .map(|(nw, dnw )| {
                    zip(nw, dnw).map(add).collect()
                })
                .collect::<Vec<Vec<Vec<f32>>>>()
                .try_into().unwrap();
        }

        // What is learning_rate?
        let rate = learning_rate / (mini_batch.len() as f32);

        self.weights = zip(&self.weights, nabla_weights)
            .map(|(w, nw)| minuss(&w, &mull(&nw, rate)))
            .collect::<Vec<Vec<Vec<f32>>>>()
            .try_into().unwrap();

        self.biases = zip(&self.biases, nabla_biases)
            .map(|(b, nb)| minus(b, &mul(&nb, rate)))
            .collect::<Vec<Vec<f32>>>()
            .try_into().unwrap();
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
    /// sample from entire training set.
    fn stochastic_gradient_descent(&mut self,
        mut training_data: TrainingData,
        epochs: i32,
        mini_batch_size: usize,
        learning_rate: f32,                 // eta is learning rate η.
        test_data: Option<TrainingData>)
    {
        let mut rng = rand::thread_rng();

        for epoch in 0..epochs {

            training_data.shuffle(&mut rng);

            /*
            Divide training data into Vector of Vectors of size mini_batch_size.

            let mini_batches: Vec<Vec<(f32, f32)>> =
                (0..n)
                .step_by(mini_batch_size)
                .map(|k| training_data[k..k+mini_batch_size].try_into().unwrap())
                .collect();
            */

            let mini_batches = {
                let mut mini_batches: Vec<TrainingData> = Vec::new();
                for k in (0..training_data.len()).step_by(mini_batch_size) {
                    mini_batches.push(training_data[k..k+mini_batch_size].try_into().unwrap())
                }
                mini_batches
            };

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


    /*
    fn print(&self) {
        println!("Layers: {}", self.num_layers);
        println!("Nodes per layer: {:?}", self.sizes);
        println!("Biases:");
        println!("{:#?}", self.biases);
        println!("Weights:");
        println!("{:#?}", self.weights);
    }
    */
}



fn main() {
    let network = Network::new([3, 5, 2]);
    
    // network.print();

    let a = vec!(5.0, 4.0, 3.1);

    let result = network.feedforward(&a);

    println!("{:?}", result);

    //network.stochastic_gradient_descent();
}
