use lazy_static::lazy_static;
use rand::distributions::Distribution;
use rand::{rngs::ThreadRng, thread_rng};
use statrs::distribution::Normal;

fn main() {
    println!("Hello, world!");
    let a: HiddenLayer<10, 20, 30> = HiddenLayer::new(ActivationFunction::GELU);
}

enum LossOutputSetup {
    SquaredReLU,
    SquaredLinear,
}

impl LossOutputSetup {
    fn get(&self) -> fn(f64, f64) -> f64 {
        match self {
            LossOutputSetup::SquaredReLU => |x: f64, y: f64| 0.5 * (x - y).powi(2),
            LossOutputSetup::BinaryCE => |x: f64, y: f64| -y * x.ln() - (1f64-y)*(1f64-x).ln(),
        }
    }

    fn get_delta_error(&self) -> fn(f64, f64) -> f64 {
        match self {
            LossOutputSetup::Squared => |x: f64, y: f64| x - y,
        }
    }
}


enum ActivationFunction {
    ReLU,
    Linear,
    Sigmoid,
    GELU,
}

impl ActivationFunction {
    fn get(&self) -> fn(f64) -> f64 {
        match self {
            ActivationFunction::ReLU => |x: f64| x.max(0f64),
            ActivationFunction::Linear => |x: f64| x,
            ActivationFunction::Sigmoid => |x: f64| 1f64 / (1f64 + std::f64::consts::E.powf(-x)),
            ActivationFunction::GELU => |x: f64| {
                0.5 * x
                    * (1f64
                        + (2f64 / std::f64::consts::PI).sqrt()
                            * (x + 0.044715f64 * x.powi(3)).tanh())
            },
        }
    }

    fn get_derivative(&self) -> fn(f64) -> f64 {
        match self {
            ActivationFunction::ReLU => |x: f64| if x >= 0f64 { 1f64 } else { 0f64 },
            ActivationFunction::Linear => |_x: f64| 1f64,
            ActivationFunction::Sigmoid => |x: f64| {
                let sigmoid = |y: f64| 1f64 / (1f64 + std::f64::consts::E.powf(-y));
                sigmoid(x) * (1f64 - sigmoid(x))
            },
            // See https://mlfromscratch.com/activation-functions-explained/#gelu
            ActivationFunction::GELU => |x: f64| {
                0.5 * (0.0356774 * x.powi(3) + 0.797885 * x).tanh()
                    + (0.0535161 * x.powi(3) + 0.398942 * x)
                        * (1f64 / (0.0356774 * x.powi(3) + 0.797885 * x).cosh()).powi(2)
                    + 0.5
            },
        }
    }
}

struct NN<const N_LAYERS: usize, const N_NEURONS_PER_LAYER: usize> {
    learning_rate: f64,
    input_layer: InputLayer,
    hidden_layers: [Layer<N_NEURONS_PER_LAYER>; N_LAYERS]
    output_layer: OutputLayer,
}

/// B is the number
#[derive(Copy, Clone)]
struct Neuron<const LENGTH: usize> {
    weights_layer_before: [f64; LENGTH],
    weights_layer_after: [f64; LENGTH],
    bias: f64,
}

trait Layer<const LENGTH> {
    fn forward(&mut self, activations: [f64; LENGTH]);
    fn backwards(&mut self, delta_errors: [f64; LENGTH]) -> [f64; LENGTH];
    fn calculate_delta_errors(&self, delta_errors: [f64; LENGTH]) -> [f64; LENGTH];
}


struct HiddenLayer<const LENGTH: usize> {
    neurons: [Neuron<LENGTH>; LENGTH],
    activations: [f64; LENGTH],
    activation_function: ActivationFunction,
}

impl<const LENGTH: usize> HiddenLayer<LENGTH>
{
     fn new(
        activation_function: ActivationFunction,
    ) -> HiddenLayer<LENGTH> {
        lazy_static! {
            static ref NORMAL: Normal = Normal::new(0.0, 0.01).unwrap();
        }
        let mut rng: ThreadRng = thread_rng();
        let mut neurons = [Neuron {
            weights_layer_before: [0f64; LENGTH],
            weights_layer_after: [0f64; LENGTH],
            bias: 0f64,
        }; LENGTH];
        (0..LENGTH).for_each(|n| {
            (0..LENGTH)
                .for_each(|l| neurons[n].weights_layer_before[l] = NORMAL.sample(&mut rng))
        });
        HiddenLayer {
            neurons,
            activation_function,
            activations: [0f64; LENGTH]
        }
    }
}
impl<const LENGTH: usize>
    Layer<LENGTH> for HiddenLayer<LENGTH> {

    fn forward(&mut self, activations: [f64; LENGTH]) {
        let activation_function = self.activation_function.get();
        let mut a = [0f64; LENGTH];
        for (i, neuron) in self.neurons.iter().enumerate() {
            a[i] = activation_function(
                activations
                    .iter()
                    .zip(neuron.weights_layer_before.iter())
                    .map(|(a, w)| a * w)
                    .sum::<f64>()
                    + neuron.bias,
            );
        }
        self.activations = a;
    }

    fn calculate_delta_errors(&self, delta_errors: [f64; LENGTH]) -> [f64; LENGTH] {
        let derivative_activation_function = self.activation_function.get_derivative();

        let mut deltas = [0f64; LENGTH];

        for (i, neuron) in self.neurons.iter().enumerate() {
            deltas[i] = derivative_activation_function(
                delta_errors
                    .iter()
                    .zip(neuron.weights_layer_after.iter())
                    .map(|(a, w)| a * w)
                    .sum::<f64>(), // + neuron.bias,
            );
        }

        
        deltas
    }

    fn backwards(&mut self, delta_errors: [f64; LENGTH]) -> [f64; LENGTH] {
        let learning_rate = 0.01;
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            for (j, weight) in neuron.weights_layer_after.iter_mut().enumerate() {
                *weight -= learning_rate * delta_errors[j] * self.activations[i];
            }
        }
        self.calculate_delta_errors(delta_errors)
    }
}

struct InputLayer<const LENGTH: usize> {

}

struct OutputLayer<const LENGTH: usize>{
    neurons: [Neuron<LENGTH>; LENGTH],
    outputs: [f64; LENGTH],
    output_function: ActivationFunction,
}

impl<const LENGTH: usize>
    Layer<LENGTH> for OutputLayer<LENGTH>
{

    fn forward(&mut self, activations: [f64; LENGTH]) {
        let output_function = self.output_function.get();
        let mut a = [0f64; LENGTH];
        for (i, neuron) in self.neurons.iter().enumerate() {
            a[i] = output_function(
                activations
                    .iter()
                    .zip(neuron.weights_layer_before.iter())
                    .map(|(a, w)| a * w)
                    .sum::<f64>()
                    + neuron.bias,
            );
        }
        self.outputs = a;
    }

    fn calculate_delta_errors(&self) -> [f64; LENGTH] {
        let mut deltas = [0f64; LENGTH];

        for (i, neuron) in self.neurons.iter().enumerate() {
            deltas[i] = derivative_output_function(
                self.outputs[],
            );
        }

        
        deltas
    }

    fn backwards(&mut self) -> [f64; LENGTH] {
        let learning_rate = 0.01;
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            for (j, weight) in neuron.weights_layer_after.iter_mut().enumerate() {
                *weight -= learning_rate * delta_errors[j] * self.outputs[i];
            }
        }
        self.calculate_delta_errors(delta_errors)
    }
}

