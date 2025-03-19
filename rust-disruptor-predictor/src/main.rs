#![allow(dead_code)]

/// A simple implementation of a Disruptor-based predictor in Rust.

mod sets;
mod signals;

use sets::{Set, SetCollection, Taxonomy, Xs};
use signals::*;

fn demo_teoria() {
    let mut set_collection = SetCollection::new();
    
    let y1 = Set::new("y1".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
    let y2 = Set::new("y2".to_string(), vec![14.0, 15.0, 16.0, 17.0]);
    
    set_collection.add_set(y1);
    set_collection.add_set(y2);
    
    let xs = Xs::from(13.0);
    
    let centroids_matrix = set_collection.calculate_centroids_matrix(&xs);
    println!("La matriz de centroides es: {}", centroids_matrix.to_string(3));
    
    let tx = set_collection.taxonomize(xs);
    println!("{}", Taxonomy::vec_to_string(&tx));
    
    let prob = set_collection.pobabilize(&tx);
    println!("\nLa matriz de probabilidades es: {}", prob.to_string(3));
    
    let prob_avg = set_collection.probabilize_avg(&tx);
    println!("La probabilidad media es: {:?}", prob_avg);
}

fn resultados_excel(pattern: &str) -> Vec<(SignalFeatures, SignalFeatures)> {
    let signals = Signal::from_file_pattern(pattern);
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
        let means = &means.values;
        let stds = &stds.values;
        println!("\nCaracterísticas de la señal {}: ", signal.label);
        println!("(signal 1 excel) mean:    min {:.10?}, max: {:.10?}", means.iter().fold(f64::MAX, |a, b| a.min(*b)), means.iter().fold(f64::MIN, |a, b| a.max(*b)));
        println!("(signal 2 excel) fft std: min {:.10?}, max: {:.10?}", stds.iter().fold(f64::MAX, |a, b| a.min(*b)), stds.iter().fold(f64::MIN, |a, b| a.max(*b)));
    }
    let feat_size = (features[0].0.values.len(), features[0].1.values.len());
    
    println!("\n\n");
    println!("Longitud de las features: ({:?}, {:?})", feat_size.0, feat_size.1);
    
    assert!(feat_size.0 == feat_size.1, "Las features no tienen la misma longitud");
    let feat_size = feat_size.0 - 1;
    let feat_mean = &features[0].0.values;
    let feat_std = &features[0].1.values;
    
    println!("[0]: ({:.*?}, {:.*?})", 15, feat_mean[0], 15, feat_std[0]);
    println!("[1]: ({:.*?}, {:.*?})", 15, feat_mean[1], 15, feat_std[1]);
    println!("[2]: ({:.*?}, {:.*?})", 15, feat_mean[2], 15, feat_std[2]);
    println!("[3]: ({:.*?}, {:.*?})", 15, feat_mean[3], 15, feat_std[3]);
    println!("[4]: ({:.*?}, {:.*?})", 15, feat_mean[4], 15, feat_std[4]);
    println!("[5]: ({:.*?}, {:.*?})", 15, feat_mean[5], 15, feat_std[5]);
    println!(".");
    println!(".");
    println!(".");
    println!("[n-4]: ({:.*?}, {:.*?})", 15, feat_mean[feat_size - 4], 15, feat_std[feat_size - 4]);
    println!("[n-3]: ({:.*?}, {:.*?})", 15, feat_mean[feat_size - 3], 15, feat_std[feat_size - 3]);
    println!("[n-2]: ({:.*?}, {:.*?})", 15, feat_mean[feat_size - 2], 15, feat_std[feat_size - 2]);
    println!("[n-1]: ({:.*?}, {:.*?})", 15, feat_mean[feat_size - 1], 15, feat_std[feat_size - 1]);
    println!("[n]:   ({:.*?}, {:.*?})", 15, feat_mean[feat_size], 15,     feat_std[feat_size]);
    
    println!("\n\n");
    println!("Min features 0: ({:.*?}, {:.*?})", 15, features[1].0.values.iter().fold(f64::MAX, |a, b| a.min(*b)), 15, features[1].1.values.iter().fold(f64::MAX, |a, b| a.min(*b)));
    println!("Max features 0: ({:.*?}, {:.*?})", 15, features[1].0.values.iter().fold(f64::MIN, |a, b| a.max(*b)), 15, features[1].1.values.iter().fold(f64::MIN, |a, b| a.max(*b)));
    features
}

fn teoria(pattern: &str) {
    let signals = Signal::normalize_vec(Signal::from_file_pattern(pattern));
    let features = signals
        .iter()
        .map(|s| s.get_features(16))
        .collect::<Vec<_>>();

    let (sets_mean, sets_std): (Vec<Set>, Vec<Set>) = features
        .iter()
        .map(|(m, s)| (Set::from(m), Set::from(s)))
        .unzip();

    let mut set_collection = SetCollection::new();
    set_collection.add_sets(sets_mean);
    set_collection.add_sets(sets_std);
    println!("Se han leído {:?} señales", signals.len());
}

fn main() {
    let ficheros = "remuestreados/DES_*_01*.txt";
    
    demo_teoria();
    resultados_excel(ficheros);
    teoria(ficheros);
}
