

pub struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Matrix {
        Matrix{data}
    }

    pub fn rows_count(&self) -> usize {
        self.data.len()
    }

    pub fn cols_count(&self) -> usize {
        self.data[0].len()
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    pub fn get_vec_vec(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    pub fn to_string(&self, decimals: usize) -> String {
        let mut matrix_str = "\n\n".to_string();
        
        for row in &self.data {
            matrix_str.push_str("[ ");
            for col in row {
                matrix_str.push_str(&format!("{:.*} ", decimals, col));
            }
            matrix_str.push_str("]\n");
        }

        matrix_str
    }       
}