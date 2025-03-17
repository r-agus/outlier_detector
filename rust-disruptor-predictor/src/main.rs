/// A simple implementation of a Disruptor-based predictor in Rust.

mod sets;
mod signals;

use sets::{Set, SetCollection, Taxonomy, Xs};
use signals::*;

fn main() {
    let mut set_collection = SetCollection::new();
    
    let y1 = Set::new("y1".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
    let y2 = Set::new("y2".to_string(), vec![14.0, 15.0, 16.0, 17.0]);
    
    set_collection.add_set(y1);
    set_collection.add_set(y2);

    let xs = Xs { value: 13.0 };

    let centroids_matrix = set_collection.calculate_centroids_matrix(&xs);
    println!("La matriz de centroides es: {}", centroids_matrix.to_string(3));

    let tx = set_collection.taxonomize(xs);
    println!("{}", Taxonomy::vec_to_string(&tx));

    let prob = set_collection.pobabilize(&tx);
    println!("\nLa matriz de probabilidades es: {}", prob.to_string(3));

    let prob_avg = set_collection.probabilize_avg(&tx);
    println!("La probabilidad media es: {:?}", prob_avg);

    let signals = Signal::from_file_pattern("remuestreados/DES_*_01*.txt");
    println!("Se han leído {:?} señales", signals.len());

    // signals.iter_mut().for_each(|s| s._normalize());
    for s in &signals {
        println!("Señal: {:?}, min: {:?}, max: {:?}, muestras: {}", s.label, s.min(), s.max(), s.values.len());
    }

    let signals = Signal::normalize_vec(signals);
    let features = signals
        .iter()
        .map(|s| s.get_features(16))
        .collect::<Vec<_>>();

    println!("\n\nDespués de normalizar:");
    for s in &signals {
        println!("Señal: {:?}, min: {:?}, max: {:?}", s.label, s.min(), s.max());
    }

    for ((means, stds), signal) in features.iter().zip(signals) {
        println!("\nCaracterísticas de la señal {}: ", signal.label);
        println!("std: min {:.10?}, max: {:.10?}", stds.iter().fold(f64::MAX, |a, b| a.min(*b)), stds.iter().fold(f64::MIN, |a, b| a.max(*b)));
        println!("mean: min {:.10?}, max: {:.10?}", means.iter().fold(f64::MAX, |a, b| a.min(*b)), means.iter().fold(f64::MIN, |a, b| a.max(*b)));
    }

    println!("\n\n");
    println!("Longitud de las features: ({:?}, {:?})", features[0].0.len(), features[0].1.len());
    println!("[0]: ({:.*?}, {:.*?})", 15, features[0].0[0], 15, features[0].1[0]);
    println!("[1]: ({:.*?}, {:.*?})", 15, features[0].0[1], 15, features[0].1[1]);
    println!("[2]: ({:.*?}, {:.*?})", 15, features[0].0[2], 15, features[0].1[2]);
    println!(".");
    println!(".");
    println!(".");
    println!("[n-2]: ({:.*?}, {:.*?})", 15, features[0].0[features[0].0.len() - 3], 15, features[0].1[features[0].1.len() - 3]);
    println!("[n-1]: ({:.*?}, {:.*?})", 15, features[0].0[features[0].0.len() - 2], 15, features[0].1[features[0].1.len() - 2]);
    println!("[n]: ({:.*?}, {:.*?})", 15, features[0].0[features[0].0.len() - 1], 15, features[0].1[features[0].1.len() - 1]);

    println!("\n\n");
    println!("Min features 0: ({:.*?}, {:.*?})", 15, features[1].0.iter().fold(f64::MAX, |a, b| a.min(*b)), 15, features[1].1.iter().fold(f64::MAX, |a, b| a.min(*b)));
    println!("Max features 0: ({:.*?}, {:.*?})", 15, features[1].0.iter().fold(f64::MIN, |a, b| a.max(*b)), 15, features[1].1.iter().fold(f64::MIN, |a, b| a.max(*b)));
}
