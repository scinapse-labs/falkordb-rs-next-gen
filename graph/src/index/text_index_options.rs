use std::sync::Arc;

#[derive(Debug, Default, Clone)]
pub struct TextIndexOptions {
    pub weight: Option<f64>,
    pub nostem: Option<bool>,
    pub phonetic: Option<bool>,
    pub language: Option<Arc<String>>,
    pub stopwords: Option<Vec<Arc<String>>>,
}
