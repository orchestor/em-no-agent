#[derive(Clone)]
pub struct MaxwellSampleVis {
    pub xs: Vec<f32>,
    pub eps: Vec<f32>,
    pub e_true: Vec<f32>,
    pub e_pred: Vec<f32>,
    pub h_true: Vec<f32>,
    pub h_pred: Vec<f32>,
}