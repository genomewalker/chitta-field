use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::MemoryId;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorDescriptor {
    pub realm: String,
    pub task_type: String,  // coding|research|analysis|distillation|unknown
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdCell {
    pub genome_id: MemoryId,
    pub fitness: f32,
    pub n_evals: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QdArchive {
    cells: HashMap<BehaviorDescriptor, QdCell>,
}

impl QdArchive {
    pub fn new() -> Self { Self::default() }

    /// Update cell: replace if new fitness > existing, else update running mean.
    pub fn update(&mut self, desc: BehaviorDescriptor, genome_id: MemoryId, fitness: f32) {
        let cell = self.cells.entry(desc).or_insert(QdCell { genome_id, fitness, n_evals: 0 });
        cell.n_evals += 1;
        if fitness > cell.fitness {
            cell.genome_id = genome_id;
            cell.fitness = fitness;
        } else {
            // running mean for stable estimation
            cell.fitness += (fitness - cell.fitness) / cell.n_evals as f32;
        }
    }

    pub fn best_for(&self, desc: &BehaviorDescriptor) -> Option<&QdCell> {
        self.cells.get(desc)
    }

    pub fn all_cells(&self) -> &HashMap<BehaviorDescriptor, QdCell> { &self.cells }

    pub fn n_filled(&self) -> usize { self.cells.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_fitness_replaces_genome() {
        let mut a = QdArchive::new();
        let d = BehaviorDescriptor { realm: "r".into(), task_type: "coding".into() };
        a.update(d.clone(), 1, 0.5);
        a.update(d.clone(), 2, 0.9);
        let c = a.best_for(&d).unwrap();
        assert_eq!(c.genome_id, 2);
        assert_eq!(c.fitness, 0.9);
        assert_eq!(c.n_evals, 2);
        assert_eq!(a.n_filled(), 1);
    }

    #[test]
    fn lower_fitness_keeps_genome_and_updates_mean() {
        let mut a = QdArchive::new();
        let d = BehaviorDescriptor { realm: "r".into(), task_type: "coding".into() };
        a.update(d.clone(), 1, 0.8);
        a.update(d.clone(), 2, 0.4);
        let c = a.best_for(&d).unwrap();
        assert_eq!(c.genome_id, 1, "lower fitness must not replace genome");
        assert!((c.fitness - 0.6).abs() < 1e-6, "running mean of 0.8,0.4 = 0.6, got {}", c.fitness);
    }
}
