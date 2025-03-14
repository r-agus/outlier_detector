/// A simple implementation of a Disruptor-based predictor in Rust.

/// A Set is a collection of values that are associated with a label.
#[derive(Debug)]
struct Set {
    label: String,
    values: Vec<f64>,
    xs: Option<Xs>,
}

/// Xs is an special set that contains a single value.
#[derive(Clone, Debug)]
struct Xs {
    value: f64,
}

struct SetCollection {
    sets: Vec<Set>,
}

/// A Taxonomy is vector of tuples where the first element is a label and second element is a value.
/// A tuple belongs to a taxonomy if the nearest centroid of the point is the label of the taxonomy.
#[derive(Debug)]
struct Taxonomy {
    label: String,
    data: Vec<(String, f64)>,
}

impl Set {
    fn calculate_centroid(&self) -> f64 {
        match self.xs.clone() {
            Some(x) => (self.values.iter().sum::<f64>() + x.value) / (self.values.len() as f64 + 1.0),
            None => self.values.iter().sum::<f64>() / (self.values.len() as f64)
        }
    }

    fn add_xs(&mut self, xs: Xs) {
        self.xs = Some(xs);
    }

    fn remove_xs(&mut self) {
        self.xs = None;
    }
}

impl SetCollection {
    fn new() -> SetCollection {
        SetCollection{sets: Vec::new()}
    }

    fn add_set(&mut self, set: Set) {
        self.sets.push(set);
    }

    fn calculate_centroids(&mut self, xs: Xs, xs_label: String) -> Vec<f64> {
        let mut centroids = Vec::new();
        
        self.sets.iter_mut().for_each(|set| {
            if set.label == xs_label {
                set.add_xs(xs.clone());
                centroids.push(set.calculate_centroid());
                set.remove_xs();
            } else {
                centroids.push(set.calculate_centroid());
            }
        });

        centroids
    }

    fn calculate_centroids_matrix(&mut self, xs: &Xs) -> Vec<Vec<f64>> {
        let mut centroids_matrix = Vec::new();
        
        // Get all labels first
        let labels: Vec<String> = self.sets.iter().map(|set| set.label.clone()).collect();
        
        // Calculate centroids for each label
        for label in labels {
            centroids_matrix.push(self.calculate_centroids(xs.clone(), label));
        }

        centroids_matrix
    }

    /// Given a value, returns the label of the set with the closest centroid.
    fn _categorize(&self, x: &f64) -> String {
        let mut closest_set = String::new();
        let mut min_distance = f64::MAX;

        for set in self.sets.iter() {
            let centroid = set.calculate_centroid();
            let distance = (centroid - x).abs();

            if distance < min_distance {
                min_distance = distance;
                closest_set = set.label.clone();
            }
        }

        closest_set
    }

    /// Returns the taxonomies where xs is categorized, for each row in the centroids matrix.
    fn taxonomize(&mut self, xs: Xs) {
        // Por cada fila de la matriz de centroides
        // Necesito una taxonomia por cada Set (columnas de la matriz de centroides, o numero de sets)
        // Cada taxonomia se inicializa con el label del set
        // Un punto pertenece a la taxonomia con el centroide mas cercano dentro de su fila
        let centroids_matrix = self.calculate_centroids_matrix(&xs);
        
        for (i, _rows) in centroids_matrix.iter().enumerate() {
            let mut taxonomies = Vec::new();
            // Create taxonomies
            for set in &self.sets {
                let taxonomy = Taxonomy::new(set.label.clone());
                taxonomies.push(taxonomy);
            }
            
            // Centroids for categorization
            let centroids: Vec<(String, f64)> = centroids_matrix[i].iter().zip(self.sets.iter()).map(|(c, s)| (s.label.clone(), *c)).collect();
            // println!("{:?}", centroids);
            let categorize = |x: &f64| -> String {
                let mut closest_set = String::new();
                let mut min_distance = f64::MAX;
                
                for (label, centroid) in &centroids {
                    let distance = (centroid - x).abs();
                    
                    if distance < min_distance {
                        min_distance = distance;
                        closest_set = label.clone();
                    }
                }
                
                closest_set
            };
                
            // Categorizations
            let mut categorized_values = Vec::new();

            for (j, set) in &mut self.sets.iter_mut().enumerate() {
                if i == j {
                    let x = Xs{value: xs.value.clone()};
                    set.add_xs(x);
                } else {
                    set.remove_xs();
                }
                // println!("{:?}", set);
                for &v in &set.values {
                    let category = categorize(&v);
                    categorized_values.push((set.label.clone(), v, category));
                }

                if set.xs.is_some() {
                    let category = categorize(&set.xs.as_ref().unwrap().value);
                    categorized_values.push((set.label.clone(), set.xs.as_ref().unwrap().value, category.clone()));

                    // Aqui sabemos a que taxonomia pertenece el punto xs
                    println!("Xs belongs to {:?}", category); 

                    // Podemos eliminar las taxonomias que no contienen el punto xs
                    taxonomies.retain(|t| t.label == category);

                    set.remove_xs();
                }
            }
            
            // Assign data to taxonomies
            for mut tx in taxonomies {
                for (set_label, value, category) in &categorized_values {
                    if &tx.label == category {
                        tx.add_data((set_label.clone(), *value));
                    }
                }
                println!("{:?}", tx);
            }
            // taxonomies.iter().filter(|t| t.data.contains(xs));
            // Tengo que mantener la taxonomia que contiene el punto xs, que coincide con la 
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
    
    let y1 = Set{label: "y1".to_string(), values: vec![5.0, 6.0, 7.0, 8.0], xs: None};
    let y2 = Set{label: "y2".to_string(), values: vec![14.0, 15.0, 16.0, 17.0], xs: None};

    set_collection.add_set(y1);
    set_collection.add_set(y2);

    let xs = Xs {value: 13.0};

    let centroids_matrix = set_collection.calculate_centroids_matrix(&xs);

    println!("{:?}", centroids_matrix);

    set_collection.taxonomize(xs);
}
