#![allow(dead_code)]

/// A simple implementation of a Disruptor-based predictor in Rust.
mod sets;
mod signals;

use sets::{Set, SetCollection, Taxonomy, Xs};
use signals::*;

fn demo_first_venn_predictor() {
    let mut set_collection = SetCollection::new();

    let y1 = Set::new("y1".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
    let y2 = Set::new("y2".to_string(), vec![14.0, 15.0, 16.0, 17.0]);

    set_collection.add_set(y1);
    set_collection.add_set(y2);

    let xs = Xs::from(13.0);

    let centroids_matrix = set_collection.calculate_centroids_matrix(&xs);
    println!(
        "La matriz de centroides es: {}",
        centroids_matrix.to_string(3)
    );

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

    // signals.iter_mut().for_each(|s| s.normalize());
    for s in &signals {
        println!(
            "Señal: {:?}, min: {:?}, max: {:?}, muestras: {}",
            s.label,
            s.min(),
            s.max(),
            s.values.len()
        );
    }

    let signals = Signal::normalize_vec(signals);
    let features = signals
        .iter()
        .map(|s| s.get_features(16))
        .collect::<Vec<_>>();

    println!("\n\nDespués de normalizar:");
    for s in &signals {
        println!(
            "Señal: {:?}, min: {:?}, max: {:?}",
            s.label,
            s.min(),
            s.max()
        );
    }

    for ((means, stds), signal) in features.iter().zip(signals) {
        let means = &means.values;
        let stds = &stds.values;
        println!("\nCaracterísticas de la señal {}: ", signal.label);
        println!(
            "(signal 1 excel) mean:    min {:.10?}, max: {:.10?}",
            means.iter().fold(f64::MAX, |a, b| a.min(*b)),
            means.iter().fold(f64::MIN, |a, b| a.max(*b))
        );
        println!(
            "(signal 2 excel) fft std: min {:.10?}, max: {:.10?}",
            stds.iter().fold(f64::MAX, |a, b| a.min(*b)),
            stds.iter().fold(f64::MIN, |a, b| a.max(*b))
        );
    }
    let feat_size = (features[0].0.values.len(), features[0].1.values.len());

    println!("\n\n");
    println!(
        "Longitud de las features: ({:?}, {:?})",
        feat_size.0, feat_size.1
    );

    assert!(
        feat_size.0 == feat_size.1,
        "Las features no tienen la misma longitud"
    );
    let feat_size = feat_size.0 - 1;
    let feat_mean = &features[0].0.values;
    let feat_std = &features[0].1.values;

    for i in 0..=5 {
        println!(
            "[{}]: ({:.*?}, {:.*?})",
            i, 15, feat_mean[i], 15, feat_std[i]
        );
    }
    println!(".");
    println!(".");
    println!(".");
    for i in (feat_size - 5)..=feat_size {
        println!(
            "[{}]: ({:.*?}, {:.*?})",
            i, 15, feat_mean[i], 15, feat_std[i]
        );
    }

    println!("\n\n");
    println!(
        "Min features 0: ({:.*?}, {:.*?})",
        15,
        features[1].0.values.iter().fold(f64::MAX, |a, b| a.min(*b)),
        15,
        features[1].1.values.iter().fold(f64::MAX, |a, b| a.min(*b))
    );
    println!(
        "Max features 0: ({:.*?}, {:.*?})",
        15,
        features[1].0.values.iter().fold(f64::MIN, |a, b| a.max(*b)),
        15,
        features[1].1.values.iter().fold(f64::MIN, |a, b| a.max(*b))
    );
    features
}

fn teoria(pattern: &str) {
    let mut set_collection = SetCollection::new();
    let signals = Signal::normalize_vec(Signal::from_file_pattern(pattern));

    let features = signals
        .iter()
        .map(|s| s.get_features(16))
        .collect::<Vec<_>>();

    let (sets_mean, _sets_std): (Vec<Set>, Vec<Set>) = features
        .iter()
        .map(|(m, s)| (Set::from(m), Set::from(s)))
        .unzip();

    set_collection.add_sets(sets_mean);
    // set_collection.add_sets(sets_std);

    let xs = signals[0].values.iter().map(|v| v * 1.1).collect::<Xs>();

    let centroids_matrix = set_collection.calculate_centroids_matrix(&xs);
    println!(
        "La matriz de centroides tiene tamaño: {:?}",
        centroids_matrix.size()
    );
    println!(
        "La matriz de centroides es: {}",
        centroids_matrix.to_string(3)
    );
}

fn main() {
    let ficheros = "remuestreados/DES_*_01*.txt";

    demo_first_venn_predictor();
    resultados_excel(ficheros);
    teoria(ficheros);
}
