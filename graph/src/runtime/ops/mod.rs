//! Runtime operator implementations for the query execution engine.
//!
//! Each operator in this module implements the [`Iterator`] trait, yielding
//! `Result<Env, String>` items. Operators are composed into a pull-based
//! execution tree via the [`OpIter`] enum, which dispatches `next()` calls
//! to the concrete operator inside each variant.
//!
//! ```text
//!                          OpIter (enum dispatch)
//!                                  |
//!          +-----------+-----------+-----------+-- ...
//!          |           |           |           |
//!     AggregateOp  FilterOp   SortOp    ProjectOp   (leaf & pipe operators)
//!          |           |
//!       child iter  child iter
//!
//! Pull model:  parent.next()  -->  child.next()  -->  ...  -->  scan.next()
//! ```
//!
//! ## Operator categories
//!
//! | Category     | Operators |
//! |--------------|-----------|
//! | **Scans**    | `NodeByLabelScan`, `NodeByIndexScan`, `NodeByIdSeek`, `NodeByLabelAndIdScan`, `NodeByFulltextScan` |
//! | **Traversal**| `CondTraverse`, `CondVarLenTraverse`, `ExpandInto` |
//! | **Filter**   | `Filter`, `Distinct`, `Skip`, `Limit`, `Sort` |
//! | **Mutation** | `Create`, `Delete`, `Set`, `Remove`, `Merge`, `Commit` |
//! | **Control**  | `Apply`, `SemiApply`, `Optional`, `OrApplyMultiplexer`, `CartesianProduct`, `Union`, `Argument` |
//! | **Transform**| `Project`, `Aggregate`, `Unwind`, `PathBuilder`, `LoadCsv`, `ProcedureCall` |
//! | **Sentinel** | `Empty` (yields nothing), `OnceOk` (yields one env) |

pub mod aggregate;
pub mod apply;
pub mod argument;
pub mod cartesian_product;
pub mod commit;
pub mod cond_traverse;
pub mod cond_var_len_traverse;
pub mod create;
pub mod delete;
pub mod distinct;
pub mod empty;
pub mod expand_into;
pub mod filter;
pub mod limit;
pub mod load_csv;
pub mod merge;
pub mod node_by_fulltext_scan;
pub mod node_by_id_seek;
pub mod node_by_index_scan;
pub mod node_by_label_and_id_scan;
pub mod node_by_label_scan;
pub mod optional;
pub mod or_apply_multiplexer;
pub mod path_builder;
pub mod procedure_call;
pub mod project;
pub mod remove;
pub mod semi_apply;
pub mod set;
pub mod skip;
pub mod sort;
pub mod union;
pub mod unwind;

pub use aggregate::AggregateOp;
pub use apply::ApplyOp;
pub use argument::ArgumentOp;
pub use cartesian_product::CartesianProductOp;
pub use commit::CommitOp;
pub use cond_traverse::CondTraverseOp;
pub use cond_var_len_traverse::CondVarLenTraverseOp;
pub use create::CreateOp;
pub use delete::DeleteOp;
pub use distinct::DistinctOp;
pub use empty::EmptyOp;
pub use expand_into::ExpandIntoOp;
pub use filter::FilterOp;
pub use limit::LimitOp;
pub use load_csv::LoadCsvOp;
pub use merge::MergeOp;
pub use node_by_fulltext_scan::NodeByFulltextScanOp;
pub use node_by_id_seek::NodeByIdSeekOp;
pub use node_by_index_scan::NodeByIndexScanOp;
pub use node_by_label_and_id_scan::NodeByLabelAndIdScanOp;
pub use node_by_label_scan::NodeByLabelScanOp;
pub use optional::OptionalOp;
pub use or_apply_multiplexer::OrApplyMultiplexerOp;
pub use path_builder::PathBuilderOp;
pub use procedure_call::ProcedureCallOp;
pub use project::ProjectOp;
pub use remove::RemoveOp;
pub use semi_apply::SemiApplyOp;
pub use set::SetOp;
pub use skip::SkipOp;
pub use sort::SortOp;
pub use union::UnionOp;
pub use unwind::UnwindOp;

use crate::runtime::env::Env;

pub enum OpIter<'a> {
    Empty(EmptyOp),
    Argument(ArgumentOp),
    Aggregate(AggregateOp<'a>),
    Apply(ApplyOp<'a>),
    CartesianProduct(CartesianProductOp<'a>),
    Commit(CommitOp<'a>),
    CondTraverse(CondTraverseOp<'a>),
    CondVarLenTraverse(CondVarLenTraverseOp<'a>),
    Create(CreateOp<'a>),
    Delete(DeleteOp<'a>),
    Distinct(DistinctOp<'a>),
    ExpandInto(ExpandIntoOp<'a>),
    Filter(FilterOp<'a>),
    Limit(LimitOp<'a>),
    LoadCsv(LoadCsvOp<'a>),
    Merge(MergeOp<'a>),
    NodeByFulltextScan(NodeByFulltextScanOp<'a>),
    NodeByIdSeek(NodeByIdSeekOp<'a>),
    NodeByIndexScan(NodeByIndexScanOp<'a>),
    NodeByLabelAndIdScan(NodeByLabelAndIdScanOp<'a>),
    NodeByLabelScan(NodeByLabelScanOp<'a>),
    Optional(OptionalOp<'a>),
    OrApplyMultiplexer(OrApplyMultiplexerOp<'a>),
    PathBuilder(PathBuilderOp<'a>),
    ProcedureCall(ProcedureCallOp<'a>),
    Project(ProjectOp<'a>),
    Remove(RemoveOp<'a>),
    SemiApply(SemiApplyOp<'a>),
    Set(SetOp<'a>),
    Skip(SkipOp<'a>),
    Sort(SortOp<'a>),
    Union(UnionOp<'a>),
    Unwind(UnwindOp<'a>),
    OnceOk(Option<Env>),
}

impl<'a> OpIter<'a> {
    pub fn set_argument_env(
        &mut self,
        env: &Env,
    ) {
        match self {
            Self::Argument(op) => op.env = Some(env.clone()),
            Self::Empty(_) | Self::OnceOk(_) => {}
            // These consume iter in new(), no child to recurse into
            Self::Aggregate(_) | Self::Sort(_) | Self::ProcedureCall(_) | Self::Commit(_) => {}
            // Recurse into child iter
            Self::Apply(op) => op.iter.set_argument_env(env),
            Self::CartesianProduct(op) => {
                op.argument_env = Some(env.clone());
                op.iter.set_argument_env(env);
            }
            Self::CondTraverse(op) => op.iter.set_argument_env(env),
            Self::CondVarLenTraverse(op) => op.iter.set_argument_env(env),
            Self::Create(op) => op.iter.set_argument_env(env),
            Self::Delete(op) => op.iter.set_argument_env(env),
            Self::Distinct(op) => op.iter.set_argument_env(env),
            Self::ExpandInto(op) => op.iter.set_argument_env(env),
            Self::Filter(op) => op.iter.set_argument_env(env),
            Self::Limit(op) => op.iter.set_argument_env(env),
            Self::LoadCsv(op) => op.iter.set_argument_env(env),
            Self::Merge(op) => op.iter.set_argument_env(env),
            Self::NodeByFulltextScan(op) => op.iter.set_argument_env(env),
            Self::NodeByIdSeek(op) => op.iter.set_argument_env(env),
            Self::NodeByIndexScan(op) => op.iter.set_argument_env(env),
            Self::NodeByLabelAndIdScan(op) => op.iter.set_argument_env(env),
            Self::NodeByLabelScan(op) => op.iter.set_argument_env(env),
            Self::Optional(op) => op.iter.set_argument_env(env),
            Self::OrApplyMultiplexer(op) => op.iter.set_argument_env(env),
            Self::PathBuilder(op) => op.iter.set_argument_env(env),
            Self::Project(op) => op.iter.set_argument_env(env),
            Self::Remove(op) => op.iter.set_argument_env(env),
            Self::SemiApply(op) => op.iter.set_argument_env(env),
            Self::Set(op) => op.iter.set_argument_env(env),
            Self::Skip(op) => op.iter.set_argument_env(env),
            Self::Unwind(op) => op.iter.set_argument_env(env),
            Self::Union(_) => {}
        }
    }
}

impl Iterator for OpIter<'_> {
    type Item = Result<Env, String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty(op) => op.next(),
            Self::Argument(op) => op.next(),
            Self::Aggregate(op) => op.next(),
            Self::Apply(op) => op.next(),
            Self::CartesianProduct(op) => op.next(),
            Self::Commit(op) => op.next(),
            Self::CondTraverse(op) => op.next(),
            Self::CondVarLenTraverse(op) => op.next(),
            Self::Create(op) => op.next(),
            Self::Delete(op) => op.next(),
            Self::Distinct(op) => op.next(),
            Self::ExpandInto(op) => op.next(),
            Self::Filter(op) => op.next(),
            Self::Limit(op) => op.next(),
            Self::LoadCsv(op) => op.next(),
            Self::Merge(op) => op.next(),
            Self::NodeByFulltextScan(op) => op.next(),
            Self::NodeByIdSeek(op) => op.next(),
            Self::NodeByIndexScan(op) => op.next(),
            Self::NodeByLabelAndIdScan(op) => op.next(),
            Self::NodeByLabelScan(op) => op.next(),
            Self::Optional(op) => op.next(),
            Self::OrApplyMultiplexer(op) => op.next(),
            Self::PathBuilder(op) => op.next(),
            Self::ProcedureCall(op) => op.next(),
            Self::Project(op) => op.next(),
            Self::Remove(op) => op.next(),
            Self::SemiApply(op) => op.next(),
            Self::Set(op) => op.next(),
            Self::Skip(op) => op.next(),
            Self::Sort(op) => op.next(),
            Self::Union(op) => op.next(),
            Self::Unwind(op) => op.next(),
            Self::OnceOk(env) => env.take().map(Ok),
        }
    }
}
