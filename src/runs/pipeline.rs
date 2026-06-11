//! Pipeline builder + build-time validation.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::error::PipelineValidationError;
use super::model::{
    BundleId, BundleTemplate, EdgeTemplate, JobId, JobTemplate, MergePolicy, PipelineId,
    PipelineSpec,
};

/// Fluent builder for a [`PipelineSpec`].
///
/// Calling [`PipelineBuilder::build`] runs validation and returns an immutable spec.
pub struct PipelineBuilder {
    id: PipelineId,
    name: String,
    jobs: Vec<JobTemplate>,
    edges: Vec<EdgeTemplate>,
    bundles: Vec<BundleTemplate>,
    /// Optional registry of known kinds; when set, validation rejects unknown kinds.
    known_kinds: Option<HashSet<String>>,
}

impl PipelineBuilder {
    /// New builder with a fresh pipeline id.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            id: PipelineId::new(),
            name: name.into(),
            jobs: Vec::new(),
            edges: Vec::new(),
            bundles: Vec::new(),
            known_kinds: None,
        }
    }

    /// Override the pipeline id (useful for stable ids across processes).
    pub fn with_id(mut self, id: PipelineId) -> Self {
        self.id = id;
        self
    }

    /// Restrict allowed `kind` strings to a known set. Validation rejects any
    /// job whose kind is missing from this set.
    pub fn with_known_kinds<I, S>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.known_kinds = Some(kinds.into_iter().map(Into::into).collect());
        self
    }

    /// Add a job template. Returns the assigned id.
    pub fn job(mut self, template: JobTemplate) -> Self {
        self.jobs.push(template);
        self
    }

    /// Add an edge template.
    pub fn edge(mut self, template: EdgeTemplate) -> Self {
        self.edges.push(template);
        self
    }

    /// Add a bundle template.
    pub fn bundle(mut self, template: BundleTemplate) -> Self {
        self.bundles.push(template);
        self
    }

    /// Run all build-time checks and consume the builder into a [`PipelineSpec`].
    pub fn build(self) -> Result<PipelineSpec, PipelineValidationError> {
        let Self {
            id,
            name,
            jobs,
            edges,
            bundles,
            known_kinds,
        } = self;

        // -- duplicates --
        let mut seen_jobs = HashSet::new();
        for j in &jobs {
            if !seen_jobs.insert(j.id) {
                return Err(PipelineValidationError::DuplicateJob(j.id));
            }
        }
        let mut seen_edges = HashSet::new();
        for e in &edges {
            if !seen_edges.insert(e.id) {
                return Err(PipelineValidationError::DuplicateEdge(e.id));
            }
        }
        let mut seen_bundles = HashSet::new();
        for b in &bundles {
            if !seen_bundles.insert(b.id) {
                return Err(PipelineValidationError::DuplicateBundle(b.id));
            }
        }

        // -- known kinds --
        if let Some(set) = &known_kinds {
            for j in &jobs {
                if !set.contains(&j.kind) {
                    return Err(PipelineValidationError::UnknownKind {
                        job_id: j.id,
                        kind: j.kind.clone(),
                    });
                }
            }
        }

        // -- edges reference existing jobs --
        let job_ids: HashSet<JobId> = jobs.iter().map(|j| j.id).collect();
        for e in &edges {
            if !job_ids.contains(&e.from) {
                return Err(PipelineValidationError::UnknownJob {
                    edge_id: e.id,
                    job_id: e.from,
                });
            }
            if !job_ids.contains(&e.to) {
                return Err(PipelineValidationError::UnknownJob {
                    edge_id: e.id,
                    job_id: e.to,
                });
            }
        }

        // -- bundles reference existing jobs + parents --
        let bundle_ids: HashSet<BundleId> = bundles.iter().map(|b| b.id).collect();
        for b in &bundles {
            for jid in &b.job_ids {
                if !job_ids.contains(jid) {
                    return Err(PipelineValidationError::BundleUnknownJob {
                        bundle_id: b.id,
                        job_id: *jid,
                    });
                }
            }
            if let Some(p) = b.parent {
                if !bundle_ids.contains(&p) {
                    return Err(PipelineValidationError::BundleUnknownParent {
                        bundle_id: b.id,
                        parent: p,
                    });
                }
            }
            for s in &b.successor_ids {
                if !bundle_ids.contains(s) {
                    return Err(PipelineValidationError::BundleUnknownParent {
                        bundle_id: b.id,
                        parent: *s,
                    });
                }
            }
        }

        // -- same-target edge conflicts --
        // Bucket by (to, target). All edges in a bucket must agree on MergePolicy,
        // and any bucket with > 1 entry rejects MergePolicy::Reject.
        let mut buckets: HashMap<(JobId, &str), Vec<&EdgeTemplate>> = HashMap::new();
        for e in &edges {
            buckets
                .entry((e.to, e.target.as_str()))
                .or_default()
                .push(e);
        }
        for ((to, target), bucket) in &buckets {
            if bucket.len() < 2 {
                continue;
            }
            let policies: HashSet<MergePolicy> = bucket.iter().map(|e| e.merge).collect();
            if policies.len() > 1 {
                return Err(PipelineValidationError::EdgeConflict {
                    to: *to,
                    target: (*target).to_string(),
                    detail: format!(
                        "{} edges share this target with disagreeing MergePolicy values: {:?}",
                        bucket.len(),
                        policies
                    ),
                });
            }
            if policies.contains(&MergePolicy::Reject) {
                return Err(PipelineValidationError::EdgeConflict {
                    to: *to,
                    target: (*target).to_string(),
                    detail: format!(
                        "{} edges share this target with MergePolicy::Reject",
                        bucket.len()
                    ),
                });
            }
        }

        // -- DAG cycle check (DFS three-colour) --
        let mut adjacency: BTreeMap<JobId, Vec<JobId>> = BTreeMap::new();
        for j in &jobs {
            adjacency.entry(j.id).or_default();
        }
        for e in &edges {
            adjacency.entry(e.from).or_default().push(e.to);
        }
        if let Some(cycle) = find_cycle(&adjacency) {
            return Err(PipelineValidationError::Cycle(cycle));
        }

        Ok(PipelineSpec {
            id,
            name,
            jobs,
            edges,
            bundles,
        })
    }
}

/// Returns one cycle (job ids in order along the cycle) if any exists.
fn find_cycle(adj: &BTreeMap<JobId, Vec<JobId>>) -> Option<Vec<JobId>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Grey,
        Black,
    }
    let mut color: HashMap<JobId, Color> = adj.keys().map(|k| (*k, Color::White)).collect();
    let mut parent: HashMap<JobId, Option<JobId>> = adj.keys().map(|k| (*k, None)).collect();

    fn dfs(
        node: JobId,
        adj: &BTreeMap<JobId, Vec<JobId>>,
        color: &mut HashMap<JobId, Color>,
        parent: &mut HashMap<JobId, Option<JobId>>,
    ) -> Option<Vec<JobId>> {
        color.insert(node, Color::Grey);
        if let Some(neighbours) = adj.get(&node) {
            for &nx in neighbours {
                match color[&nx] {
                    Color::Grey => {
                        // walk back from `node` until we hit `nx`
                        let mut cycle = vec![nx, node];
                        let mut cur = parent[&node];
                        while let Some(p) = cur {
                            if p == nx {
                                break;
                            }
                            cycle.push(p);
                            cur = parent[&p];
                        }
                        cycle.reverse();
                        return Some(cycle);
                    }
                    Color::White => {
                        parent.insert(nx, Some(node));
                        if let Some(c) = dfs(nx, adj, color, parent) {
                            return Some(c);
                        }
                    }
                    Color::Black => {}
                }
            }
        }
        color.insert(node, Color::Black);
        None
    }

    let nodes: Vec<JobId> = adj.keys().copied().collect();
    for n in nodes {
        if color[&n] == Color::White {
            if let Some(c) = dfs(n, adj, &mut color, &mut parent) {
                return Some(c);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::model::{
        EdgeCondition, EdgeId, EdgeTemplate, JobTemplate, JoinPolicy, MergePolicy, OutputProjection,
    };

    fn job(kind: &str) -> JobTemplate {
        JobTemplate {
            id: JobId::new(),
            kind: kind.into(),
            default_inputs: serde_json::Value::Null,
            bundle_id: None,
            join_policy: JoinPolicy::AllRequired,
        }
    }

    fn edge(from: JobId, to: JobId, target: &str, merge: MergePolicy) -> EdgeTemplate {
        EdgeTemplate {
            id: EdgeId::new(),
            from,
            to,
            source: OutputProjection::Whole,
            target: target.into(),
            condition: EdgeCondition::Always,
            merge,
            required: true,
        }
    }

    #[test]
    fn cycle_is_detected() {
        let a = job("a");
        let b = job("b");
        let e1 = edge(a.id, b.id, "x", MergePolicy::LastWriteWins);
        let e2 = edge(b.id, a.id, "y", MergePolicy::LastWriteWins);
        let res = PipelineBuilder::new("p")
            .job(a)
            .job(b)
            .edge(e1)
            .edge(e2)
            .build();
        assert!(matches!(res, Err(PipelineValidationError::Cycle(_))));
    }

    #[test]
    fn duplicate_job_id_fails() {
        let a = job("a");
        let b = JobTemplate {
            id: a.id,
            ..job("b")
        };
        let res = PipelineBuilder::new("p").job(a).job(b).build();
        assert!(matches!(res, Err(PipelineValidationError::DuplicateJob(_))));
    }

    #[test]
    fn unknown_kind_fails() {
        let a = job("foo");
        let res = PipelineBuilder::new("p")
            .with_known_kinds(["bar"])
            .job(a)
            .build();
        assert!(matches!(
            res,
            Err(PipelineValidationError::UnknownKind { .. })
        ));
    }

    #[test]
    fn merge_reject_with_two_edges_fails() {
        let a = job("a");
        let b = job("b");
        let c = job("c");
        let e1 = edge(a.id, c.id, "x", MergePolicy::Reject);
        let e2 = edge(b.id, c.id, "x", MergePolicy::Reject);
        let res = PipelineBuilder::new("p")
            .job(a)
            .job(b)
            .job(c)
            .edge(e1)
            .edge(e2)
            .build();
        assert!(matches!(
            res,
            Err(PipelineValidationError::EdgeConflict { .. })
        ));
    }

    #[test]
    fn merge_disagreement_fails() {
        let a = job("a");
        let b = job("b");
        let c = job("c");
        let e1 = edge(a.id, c.id, "x", MergePolicy::AppendArray);
        let e2 = edge(b.id, c.id, "x", MergePolicy::LastWriteWins);
        let res = PipelineBuilder::new("p")
            .job(a)
            .job(b)
            .job(c)
            .edge(e1)
            .edge(e2)
            .build();
        assert!(matches!(
            res,
            Err(PipelineValidationError::EdgeConflict { .. })
        ));
    }

    #[test]
    fn diamond_dag_builds() {
        let a = job("a");
        let b = job("b");
        let c = job("c");
        let d = job("d");
        let edges = [
            edge(a.id, b.id, "x", MergePolicy::LastWriteWins),
            edge(a.id, c.id, "x", MergePolicy::LastWriteWins),
            edge(b.id, d.id, "left", MergePolicy::LastWriteWins),
            edge(c.id, d.id, "right", MergePolicy::LastWriteWins),
        ];
        let mut b_ = PipelineBuilder::new("p").job(a).job(b).job(c).job(d);
        for e in edges {
            b_ = b_.edge(e);
        }
        b_.build().expect("diamond builds");
    }
}
