/// A simple implementation of a Disruptor-based predictor in Rust.

mod sets;

use sets::{Set, SetCollection, Xs};

fn main() {
    let mut set_collection = SetCollection::new();
    
    let y1 = Set::new("y1".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
    let y2 = Set::new("y2".to_string(), vec![14.0, 15.0, 16.0, 17.0]);

    set_collection.add_set(y1);
    set_collection.add_set(y2);

    let xs = Xs {value: 13.0};

    let centroids_matrix = set_collection.calculate_centroids_matrix(&xs);

    println!("La matriz de centroides es: {:?}", centroids_matrix);

    let tx = set_collection.taxonomize(xs);
    println!("{:?}", tx);

    let prob = set_collection.pobabilize(&tx);
    println!("La matriz de probabilidades es: {:?}", prob);

    let prob_avg = set_collection.probabilize_avg(&tx);
    println!("La probabilidad media es: {:?}", prob_avg);
}
