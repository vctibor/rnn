use std::{fs::File, io::{Cursor, Read}};
use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;

/// # mnist_loader
/// 
/// A library to load the MNIST image data.  For details of the data
/// structures that are returned, see the doc strings for ``load_data``
/// and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
/// function usually called by our neural network code.
///
/// This code is combination of mnist_loader from the book and code from
/// https://ngoldbaum.github.io/posts/loading-mnist-data-in-rust/.


/// To represent the data as they exist on-disk I defined a struct named MnistData that wraps a vector containing the dimensions of the data and then a Vec<u8> that contains a flattened representation of the data.
#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

/// Now to convert the data as loaded in directly from the idx file we need to do a bit of data munging. I decided to create another struct, MnistImage that has a copy of the image vector and the classification for the image.
#[derive(Debug)]
pub struct MnistImage {
    pub image: Vec<f64>,
    pub classification: u8,
}

pub fn load_data(dataset_name: &str) -> Vec<MnistImage>
{
    let filename = format!("data/{}-labels-idx1-ubyte.gz", dataset_name);

    let label_data = &MnistData::new(&(File::open(filename)).unwrap()).unwrap();

    let filename = format!("data/{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename)).unwrap()).unwrap();
    
    let mut images: Vec<Vec<f64>> = Vec::new();
    
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();

        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            classification,
        })
    }

    ret
}


/*
/// This ought to be vector consisting of values from 0 to 1, of the same size
/// as the first (input) layer of the network.
//pub type NetworkInput = Vec<f32>;


//pub type TrainingData = 

/// Return the MNIST data as a tuple containing the training data,
/// the validation data, and the test data.
/// 
/// The ``training_data`` is returned as a tuple with two entries.
/// The first entry contains the actual training images.  This is a
/// numpy ndarray with 50,000 entries.  Each entry is, in turn, a
/// numpy ndarray with 784 values, representing the 28 * 28 = 784
/// pixels in a single MNIST image.
/// 
/// The second entry in the ``training_data`` tuple is a numpy ndarray
/// containing 50,000 entries.  Those entries are just the digit
/// values (0...9) for the corresponding images contained in the first
/// entry of the tuple.
/// 
/// The ``validation_data`` and ``test_data`` are similar, except
/// each contains only 10,000 images.
/// 
/// This is a nice data format, but for use in neural networks it's
/// helpful to modify the format of the ``training_data`` a little.
/// That's done in the wrapper function ``load_data_wrapper()``, see
/// below.
pub fn load_data() {
    
}
*/