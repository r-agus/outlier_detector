use std::collections::HashMap;

/// A simple implementation of a Disruptor-based predictor in Rust.

/// A Set is a collection of values that are associated with a label.
struct Set {
    label: String,
    values: Vec<f64>,
}

/// Xs is an special set that contains a single value.
#[derive(Clone, Copy)]
struct Xs {
    value: f64,
}

struct SetCollection {
    sets: Vec<Set>,
}

/// A Taxonomy is vector of tuples where the first element is a label and second element is a value.
/// A tuple belongs to a taxonomy if the nearest centroid of the point is the label of the taxonomy.
struct Taxonomy {
    label: String,
    data: Vec<(String, f64)>,
}

impl Set {
    fn calculate_centroid(&self, xs: Option<f64>) -> f64 {
        match xs {
            Some(x) => (self.values.iter().sum::<f64>() + x) / (self.values.len() as f64 + 1.0),
            None => self.values.iter().sum::<f64>() / (self.values.len() as f64)
        }
    }
}

impl SetCollection {
    fn new() -> SetCollection {
        SetCollection{sets: Vec::new()}
    }

    fn add_set(&mut self, set: Set) {
        self.sets.push(set);
    }

    fn calculate_centroids(&self, xs: Xs, xs_label: String) -> Vec<f64> {
        let mut centroids = Vec::new();
        
        self.sets.iter().for_each(|set| {
            if set.label == xs_label {
                centroids.push(set.calculate_centroid(Some(xs.value)));
            } else {
                centroids.push(set.calculate_centroid(None));
            }
        });

        centroids
    }

    fn calculate_centroids_matrix(&self, xs: Xs) -> Vec<Vec<f64>> {
        let mut centroids_matrix = Vec::new();

        self.sets.iter().for_each(|set| {
            centroids_matrix.push(self.calculate_centroids(xs, set.label.clone()));
        });

        centroids_matrix
    }

    /// Given a value, returns the label of the set with the closest centroid.
    fn categorize(&self, x: &f64) -> String {
        let mut closest_set = String::new();
        let mut min_distance = f64::MAX;

        for set in self.sets.iter() {
            let centroid = set.calculate_centroid(None);
            let distance = (centroid - x).abs();

            if distance < min_distance {
                min_distance = distance;
                closest_set = set.label.clone();
            }
        }

        closest_set
    }

    /// Returns the taxonomies where xs is categorized, for each row in the centroids matrix.
    fn taxonomize(&self, xs: Xs) {
        // Por cada fila de la matriz de centroides
        // Necesito una taxonomia por cada Set (columnas de la matriz de centroides, o numero de sets)
        // Cada taxonomia se inicializa con el label del set
        // Un punto pertenece a la taxonomia con el centroide mas cercano dentro de su fila
        for _rows in self.calculate_centroids_matrix(xs).iter() {
            let mut taxonomies = Vec::new();
            for set in self.sets.iter() {
                let taxonomy = Taxonomy::new(set.label.clone());
                taxonomies.push(taxonomy);
            }
            for mut tx in taxonomies {
                for set in self.sets.iter() {
                    for v in set.values.iter() {
                        if self.categorize(v) == tx.label {
                            tx.add_data((set.label.clone(), *v));
                        }
                    }
                }
            }
            // taxonomies.iter().filter(|t| t.data.contains(xs));
        }
    }
}

impl Taxonomy {
    fn new(label: String) -> Taxonomy {
        Taxonomy{label, data: Vec::new()}
    }

    fn add_data(&mut self, data: (String, f64)) {
        self.data.push(data);
    }
}

fn main() {
    let mut set_collection = SetCollection::new();
    
    let y1 = Set{label: "y1".to_string(), values: vec![5.0, 6.0, 7.0, 8.0]};
    let y2 = Set{label: "y2".to_string(), values: vec![14.0, 15.0, 16.0, 17.0]};

    set_collection.add_set(y1);
    set_collection.add_set(y2);

    let xs = Xs {value: 13.0};

    let centroids_matrix = set_collection.calculate_centroids_matrix(xs);

    println!("{:?}", centroids_matrix);

    set_collection.taxonomize(xs);
}
