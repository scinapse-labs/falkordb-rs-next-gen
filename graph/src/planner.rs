//! Query plan generation from bound AST.
//!
//! The planner converts a bound Cypher AST into a logical execution plan (IR tree).
//! This phase determines the order of operations and which algorithms to use for
//! pattern matching.
//!
//! ## Plan Structure
//!
//! The plan is a tree where:
//! - Leaf nodes produce tuples (scans, argument)
//! - Internal nodes transform/filter tuples from children
//! - The root produces the final result
//!
//! ## Key Planning Decisions
//!
//! 1. **Scan selection**: Chooses between label scans, index scans, or ID lookups
//! 2. **Join ordering**: Determines order of pattern matching for efficiency
//! 3. **Projection placement**: Decides when to project/aggregate
//! 4. **Filter pushdown**: Places filters as early as possible
//!
//! ## IR Operators
//!
//! - **NodeByLabelScan**: Scan all nodes with a label
//! - **NodeByIndexScan**: Use an index for node lookup
//! - **CondTraverse**: Traverse relationships conditionally
//! - **ExpandInto**: Check for relationship between known nodes
//! - **Filter**: Apply predicate to filter tuples
//! - **Project**: Compute new values from existing
//! - **Aggregate**: Group and aggregate tuples
//! - **Sort/Skip/Limit**: Order and paginate results

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    sync::Arc,
};

use crate::runtime::functions::Type;

use orx_tree::{Bfs, DynNode, DynTree, NodeRef, Side, Traversal, Traverser};

use crate::{
    ast::{
        BoundQueryIR, ExprIR, QueryExpr, QueryGraph, QueryIR, QueryNode, QueryPath,
        QueryRelationship, SetItem, SupportAggregation, Variable,
    },
    indexer::{EntityType, IndexQuery, IndexType},
    runtime::functions::GraphFn,
    tree,
};

/// Intermediate Representation (IR) for execution plan operators.
///
/// Each variant represents a physical operation in the query execution plan.
/// The plan forms a tree where data flows from leaves to root.
#[derive(Clone, Debug)]
pub enum IR {
    /// Empty result set (used as placeholder)
    Empty,
    /// Receives input from parent operator
    Argument,
    /// OPTIONAL MATCH - returns nulls if no match
    Optional(Vec<Variable>),
    /// CALL procedure with arguments, yielding outputs
    ProcedureCall(Arc<GraphFn>, Vec<QueryExpr<Variable>>, Vec<Variable>),
    /// UNWIND list AS variable
    Unwind(QueryExpr<Variable>, Variable),
    /// CREATE pattern
    Create(QueryGraph<Arc<String>, Arc<String>, Variable>),
    /// MERGE pattern with ON CREATE/ON MATCH actions
    Merge(
        QueryGraph<Arc<String>, Arc<String>, Variable>,
        Vec<SetItem<Arc<String>, Variable>>,
        Vec<SetItem<Arc<String>, Variable>>,
    ),
    /// DELETE entities (detach flag for relationships)
    Delete(Vec<QueryExpr<Variable>>, bool),
    /// SET properties/labels
    Set(Vec<SetItem<Arc<String>, Variable>>),
    /// REMOVE properties/labels
    Remove(Vec<QueryExpr<Variable>>),
    /// Scan all nodes (no label filter)
    AllNodeScan(Arc<QueryNode<Arc<String>, Variable>>),
    /// Scan nodes by label
    NodeByLabelScan(Arc<QueryNode<Arc<String>, Variable>>),
    /// Scan nodes using an index
    NodeByIndexScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        index: Arc<String>,
        query: Arc<IndexQuery<QueryExpr<Variable>>>,
    },
    /// Lookup node by label and id
    NodeByLabelAndIdScan {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        filter: Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    },
    /// Lookup node by id only
    NodeByIdSeek {
        node: Arc<QueryNode<Arc<String>, Variable>>,
        filter: Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    },
    /// Traverse relationships from known nodes
    CondTraverse(Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>),
    /// Variable-length traversal (BFS) from known nodes
    CondVarLenTraverse(Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>),
    /// Check relationship between two known nodes
    ExpandInto(Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>),
    /// Build path objects from matched patterns
    PathBuilder(Vec<Arc<QueryPath<Variable>>>),
    /// Apply filter predicate
    Filter(QueryExpr<Variable>),
    /// Cartesian product of child results
    CartesianProduct,
    /// Apply = correlated join: for each row from child 0, run child 1
    Apply,
    /// Semi-join: passes through left row when right produces at least one result
    SemiApply,
    /// Anti-semi-join: passes through left row when right produces NO results
    AntiSemiApply,
    /// Or-apply-multiplexer: for each row from child 0 (bound branch),
    /// test condition branches (children 1..N) with short-circuit OR semantics.
    /// Passes through the row if ANY branch succeeds.
    /// `Vec<bool>` has one entry per condition branch: true means invert the
    /// branch result (anti-semi-join semantics for NOT-pattern predicates).
    /// Scalar filter branches are placed before pattern branches for efficiency.
    OrApplyMultiplexer(Vec<bool>),
    /// Load CSV file
    LoadCsv {
        file_path: QueryExpr<Variable>,
        headers: bool,
        delimiter: QueryExpr<Variable>,
        var: Variable,
    },
    /// Sort by expressions (bool = descending)
    Sort(Vec<(QueryExpr<Variable>, bool)>),
    /// Skip first N rows
    Skip(QueryExpr<Variable>),
    /// Limit to N rows
    Limit(QueryExpr<Variable>),
    /// Aggregate with grouping keys, aggregations, and projections
    Aggregate(
        Vec<Variable>,
        Vec<(Variable, QueryExpr<Variable>)>,
        Vec<(Variable, QueryExpr<Variable>)>,
    ),
    /// Project expressions to new variables
    Project(
        Vec<(Variable, QueryExpr<Variable>)>,
        Vec<(Variable, Variable)>,
    ),
    /// Remove duplicate rows
    Distinct,
    /// Commit write operations to graph
    Commit,
    /// CREATE INDEX operation
    CreateIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
        options: Option<QueryExpr<Variable>>,
    },
    /// DROP INDEX operation
    DropIndex {
        label: Arc<String>,
        attrs: Vec<Arc<String>>,
        index_type: IndexType,
        entity_type: EntityType,
    },
}

#[cfg_attr(tarpaulin, skip)]
impl Display for IR {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "Empty"),
            Self::Argument => write!(f, "Argument"),
            Self::Optional(_) => write!(f, "Optional"),
            Self::ProcedureCall(_, _, _) => write!(f, "ProcedureCall"),
            Self::Unwind(_, _) => {
                write!(f, "Unwind")
            }
            Self::Create(pattern) => write!(f, "Create | {pattern}"),
            Self::Merge(pattern, _, _) => write!(f, "Merge | {pattern}"),
            Self::Delete(_, _) => write!(f, "Delete"),
            Self::Set(_) => write!(f, "Set"),
            Self::Remove(_) => write!(f, "Remove"),
            Self::AllNodeScan(node) => {
                write!(f, "All Node Scan | {node}")
            }
            Self::NodeByLabelScan(node) => {
                write!(f, "Node By Label Scan | {node}")
            }
            Self::NodeByIndexScan { node, .. } => {
                write!(f, "Node By Index Scan | {node}")
            }
            Self::NodeByLabelAndIdScan { node, .. } => {
                write!(f, "Node By Label and ID Scan | {node}")
            }
            Self::NodeByIdSeek { .. } => write!(f, "NodeByIdSeek"),
            Self::CondTraverse(rel) => write!(f, "Conditional Traverse | {rel}"),
            Self::CondVarLenTraverse(rel) => write!(f, "Variable Length Traverse | {rel}"),
            Self::ExpandInto(rel) => write!(f, "Expand Into | {rel}"),
            Self::PathBuilder(_) => write!(f, "PathBuilder"),
            Self::Filter(_) => write!(f, "Filter"),
            Self::CartesianProduct => write!(f, "Cartesian Product"),
            Self::Apply => write!(f, "Apply"),
            Self::SemiApply => write!(f, "Semi Apply"),
            Self::AntiSemiApply => write!(f, "Anti Semi Apply"),
            Self::OrApplyMultiplexer(_) => write!(f, "Or Apply Multiplexer"),
            Self::LoadCsv { .. } => write!(f, "Load CSV"),
            Self::Sort(_) => write!(f, "Sort"),
            Self::Skip(_) => write!(f, "Skip"),
            Self::Limit(_) => write!(f, "Limit"),
            Self::Aggregate(_, _, _) => write!(f, "Aggregate"),
            Self::Project(_, _) => write!(f, "Project"),
            Self::Commit => write!(f, "Commit"),
            Self::Distinct => write!(f, "Distinct"),
            Self::CreateIndex { label, attrs, .. } => {
                write!(f, "Create Index | :{label}({attrs:?})")
            }
            Self::DropIndex { label, attrs, .. } => {
                write!(f, "Drop Index | :{label}({attrs:?})")
            }
        }
    }
}

#[derive(Default)]
pub struct Planner {
    visited: HashSet<u32>,
    next_var_id: u32,
}

impl Planner {
    fn add_argument_to_leaves(tree: &mut DynTree<IR>) {
        let mut tr = Traversal.bfs().over_nodes();

        let leaves: Vec<_> = tree
            .root()
            .walk_with(&mut tr)
            .filter(|n| n.is_leaf() && !matches!(n.data(), IR::Argument))
            .map(|x| x.idx())
            .collect();

        // Add Argument node as a child to each leaf
        for leaf_idx in leaves {
            tree.node_mut(leaf_idx).push_child(IR::Argument);
        }
    }

    /// Check whether a rebuilt expression subtree references any of the given
    /// inline-pattern variable IDs.
    fn contains_inline_var(
        node: DynNode<ExprIR<Variable>>,
        inline_var_ids: &HashSet<u32>,
    ) -> bool {
        let mut tr = Traversal.bfs().over_nodes();
        node.walk_with(&mut tr)
            .any(|n| matches!(n.data(), ExprIR::Variable(v) if inline_var_ids.contains(&v.id)))
    }

    /// Build a pattern sub-plan for a graph, saving and restoring visited state.
    fn build_pattern_sub_plan(
        &mut self,
        graph: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    ) -> DynTree<IR> {
        let saved = self.visited.clone();
        let mut sub_plan = self.plan_match(graph, None);
        self.visited = saved;
        Self::add_argument_to_leaves(&mut sub_plan);
        sub_plan
    }

    /// Recursively decompose an expression (that may contain inline-pattern
    /// variables) into an IR sub-plan.  `input` is the upstream data stream.
    /// The returned plan filters rows: only rows for which the expression
    /// evaluates to true are passed through.
    fn expr_to_plan(
        &mut self,
        node: orx_tree::Node<orx_tree::Dyn<ExprIR<Variable>>>,
        inline_map: &HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        input: DynTree<IR>,
    ) -> DynTree<IR> {
        let inline_var_ids: HashSet<u32> = inline_map.keys().copied().collect();

        // Unwrap Paren nodes transparently
        if matches!(node.data(), ExprIR::Paren)
            && let Some(child) = node.get_child(0)
        {
            return self.expr_to_plan(child, inline_map, input);
        }

        // Inline-pattern variable → SemiApply (pass if pattern exists)
        if let ExprIR::Variable(v) = node.data()
            && let Some(graph) = inline_map.get(&v.id)
        {
            let sub_plan = self.build_pattern_sub_plan(graph);
            return tree!(IR::SemiApply, input, sub_plan);
        }

        // NOT(inline-pattern variable) → AntiSemiApply
        if matches!(node.data(), ExprIR::Not)
            && let Some(child) = node.get_child(0)
            && let ExprIR::Variable(v) = child.data()
            && let Some(graph) = inline_map.get(&v.id)
        {
            let sub_plan = self.build_pattern_sub_plan(graph);
            return tree!(IR::AntiSemiApply, input, sub_plan);
        }

        // Pure scalar (no inline var refs) → Filter
        if !Self::contains_inline_var(node.clone(), &inline_var_ids) {
            let expr_tree = node.clone_as_tree();
            return tree!(IR::Filter(Arc::new(expr_tree)), input);
        }

        // OR → OrApplyMultiplexer
        if matches!(node.data(), ExprIR::Or) {
            return self.or_expr_to_plan(node, inline_map, input);
        }

        // AND → chain conditions sequentially
        if matches!(node.data(), ExprIR::And) {
            return self.and_expr_to_plan(node, inline_map, input);
        }

        // NOT(complex_expr) → AntiSemiApply(input, inner_plan)
        // inner_plan passes rows when the inner expression is true,
        // so AntiSemiApply inverts: passes when inner is false.
        if matches!(node.data(), ExprIR::Not)
            && let Some(child) = node.get_child(0)
        {
            let inner = self.expr_to_plan(child, inline_map, tree!(IR::Argument));
            return tree!(IR::AntiSemiApply, input, inner);
        }

        // Fallback for other operators (XOR etc.) with inline vars:
        // shouldn't happen in practice; treat as opaque filter.
        let expr_tree = node.clone_as_tree();
        tree!(IR::Filter(Arc::new(expr_tree)), input)
    }

    /// Build an `OrApplyMultiplexer` for an OR expression.
    /// `input` becomes the bound branch (child 0).
    fn or_expr_to_plan(
        &mut self,
        or_node: DynNode<ExprIR<Variable>>,
        inline_map: &HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        input: DynTree<IR>,
    ) -> DynTree<IR> {
        let inline_var_ids: HashSet<u32> = inline_map.keys().copied().collect();

        // Collect OR children into owned trees so we can call &mut self freely.
        let child_trees: Vec<DynTree<ExprIR<Variable>>> =
            or_node.children().map(|c| c.clone_as_tree()).collect();

        // Classify branches: scalars first (cheap), then non-scalars.
        let mut scalar_branches: Vec<DynTree<IR>> = vec![];
        let mut other_branches: Vec<(DynTree<IR>, bool)> = vec![]; // (plan, is_anti)

        for child_tree in &child_trees {
            let child = child_tree.root();
            // Bare pattern variable → raw pattern sub-plan (anti=false)
            if let ExprIR::Variable(v) = child.data()
                && let Some(graph) = inline_map.get(&v.id)
            {
                let sub_plan = self.build_pattern_sub_plan(graph);
                other_branches.push((sub_plan, false));
                continue;
            }
            // NOT(pattern variable) → raw pattern sub-plan (anti=true)
            if matches!(child.data(), ExprIR::Not)
                && let Some(grandchild) = child.get_child(0)
                && let ExprIR::Variable(v) = grandchild.data()
                && let Some(graph) = inline_map.get(&v.id)
            {
                let sub_plan = self.build_pattern_sub_plan(graph);
                other_branches.push((sub_plan, true));
                continue;
            }
            // Pure scalar → Filter(expr, Argument)
            if !Self::contains_inline_var(child.clone(), &inline_var_ids) {
                let expr_tree = child.clone_as_tree();
                let branch = tree!(IR::Filter(Arc::new(expr_tree)), tree!(IR::Argument));
                scalar_branches.push(branch);
                continue;
            }
            // Complex child (AND with patterns, nested OR, etc.):
            // Recursively build a sub-plan starting from Argument.
            let branch = self.expr_to_plan(child, inline_map, tree!(IR::Argument));
            other_branches.push((branch, false));
        }

        // Assemble: bound branch first, then scalars, then others.
        let mut anti_flags = Vec::with_capacity(scalar_branches.len() + other_branches.len());
        let mut children = Vec::with_capacity(1 + scalar_branches.len() + other_branches.len());
        children.push(input);
        for branch in scalar_branches {
            anti_flags.push(false);
            children.push(branch);
        }
        for (branch, is_anti) in other_branches {
            anti_flags.push(is_anti);
            children.push(branch);
        }
        tree!(IR::OrApplyMultiplexer(anti_flags); children)
    }

    /// Build a chained plan for an AND expression.
    /// Scalars are applied first (cheap), then pattern / complex conditions.
    fn and_expr_to_plan(
        &mut self,
        and_node: DynNode<ExprIR<Variable>>,
        inline_map: &HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>>,
        input: DynTree<IR>,
    ) -> DynTree<IR> {
        let inline_var_ids: HashSet<u32> = inline_map.keys().copied().collect();

        // Collect children into owned trees for borrow-safety.
        let child_trees: Vec<DynTree<ExprIR<Variable>>> =
            and_node.children().map(|c| c.clone_as_tree()).collect();

        let mut scalar_parts: Vec<DynTree<ExprIR<Variable>>> = vec![];
        let mut non_scalar_trees: Vec<&DynTree<ExprIR<Variable>>> = vec![];

        for child_tree in &child_trees {
            let child = child_tree.root();
            if Self::contains_inline_var(child.clone(), &inline_var_ids) {
                non_scalar_trees.push(child_tree);
            } else {
                // Skip trivial Bool(true) from extractable pattern replacement
                if !matches!(child.data(), ExprIR::Bool(true)) {
                    scalar_parts.push(child.clone_as_tree());
                }
            }
        }

        let mut plan = input;

        // Apply scalar filter first (cheapest).
        if !scalar_parts.is_empty() {
            let filter_expr = if scalar_parts.len() == 1 {
                Arc::new(scalar_parts.into_iter().next().unwrap())
            } else {
                Arc::new(tree!(ExprIR::And; scalar_parts))
            };
            plan = tree!(IR::Filter(filter_expr), plan);
        }

        // Apply non-scalar conditions sequentially (each filters the stream).
        for child_tree in non_scalar_trees {
            plan = self.expr_to_plan(child_tree.root(), inline_map, plan);
        }

        plan
    }

    /// Finds the maximum variable ID used in an expression tree, including
    /// variables inside Pattern sub-graphs. Used to initialize `next_var_id`
    /// above all binder-assigned IDs.
    fn max_var_id_in_expr(expr: &DynTree<ExprIR<Variable>>) -> u32 {
        let mut max_id = 0u32;
        for idx in expr.root().indices::<Bfs>() {
            match expr.node(idx).data() {
                ExprIR::Variable(var) => max_id = max_id.max(var.id),
                ExprIR::Pattern(graph) => {
                    for var in graph.variables() {
                        max_id = max_id.max(var.id);
                    }
                }
                _ => {}
            }
        }
        max_id
    }

    fn collect_patterns_and_rebuild(
        node: DynNode<ExprIR<Variable>>,
        extractable: &mut Vec<(QueryGraph<Arc<String>, Arc<String>, Variable>, bool)>,
        inline: &mut Vec<(Variable, QueryGraph<Arc<String>, Arc<String>, Variable>)>,
        next_var_id: &mut u32,
        can_extract: bool,
    ) -> DynTree<ExprIR<Variable>> {
        match node.data() {
            ExprIR::Pattern(graph) => {
                if can_extract {
                    extractable.push((graph.clone(), false));
                    DynTree::new(ExprIR::Bool(true))
                } else {
                    let var = Variable {
                        name: None,
                        id: *next_var_id,
                        scope_id: 0,
                        ty: Type::Bool,
                    };
                    *next_var_id += 1;
                    inline.push((var.clone(), graph.clone()));
                    DynTree::new(ExprIR::Variable(var))
                }
            }
            ExprIR::Not => {
                if let Some(child) = node.get_child(0)
                    && let ExprIR::Pattern(graph) = child.data()
                {
                    if can_extract {
                        extractable.push((graph.clone(), true));
                        return DynTree::new(ExprIR::Bool(true));
                    }
                    let var = Variable {
                        name: None,
                        id: *next_var_id,
                        scope_id: 0,
                        ty: Type::Bool,
                    };
                    *next_var_id += 1;
                    inline.push((var.clone(), graph.clone()));
                    let mut new_tree = DynTree::new(ExprIR::Not);
                    new_tree
                        .root_mut()
                        .push_child_tree(DynTree::new(ExprIR::Variable(var)));
                    return new_tree;
                }
                let mut new_tree = DynTree::new(node.data().clone());
                for child in node.children() {
                    let child_tree = Self::collect_patterns_and_rebuild(
                        child,
                        extractable,
                        inline,
                        next_var_id,
                        false,
                    );
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
            ExprIR::And => {
                let mut new_tree = DynTree::new(ExprIR::And);
                for child in node.children() {
                    let child_tree = Self::collect_patterns_and_rebuild(
                        child,
                        extractable,
                        inline,
                        next_var_id,
                        can_extract,
                    );
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
            _ => {
                let mut new_tree = DynTree::new(node.data().clone());
                for child in node.children() {
                    let child_tree = Self::collect_patterns_and_rebuild(
                        child,
                        extractable,
                        inline,
                        next_var_id,
                        false,
                    );
                    new_tree.root_mut().push_child_tree(child_tree);
                }
                new_tree
            }
        }
    }

    fn plan_match(
        &mut self,
        pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
        filter: Option<QueryExpr<Variable>>,
    ) -> DynTree<IR> {
        let mut vec = vec![];
        for component in pattern.connected_components() {
            let relationships = component.relationships();
            let mut iter = relationships.into_iter();
            let Some(relationship) = iter.next() else {
                let nodes = component.nodes();
                debug_assert_eq!(nodes.len(), 1);
                let node = nodes[0].clone();
                if self.visited.contains(&node.alias.id) {
                    // Already bound - just use Argument (no scan needed)
                    vec.push(tree!(IR::Argument));
                } else {
                    let mut res = if node.labels.is_empty() {
                        tree!(IR::AllNodeScan(node.clone()))
                    } else {
                        tree!(IR::NodeByLabelScan(node.clone()))
                    };
                    self.visited.insert(node.alias.id);
                    let paths = component.paths();
                    if !paths.is_empty() {
                        res = tree!(IR::PathBuilder(paths), res);
                    }
                    vec.push(res);
                }
                continue;
            };
            let mut res = if relationship.from.alias.id == relationship.to.alias.id {
                let scan = if relationship.from.labels.is_empty() {
                    tree!(IR::AllNodeScan(relationship.from.clone()))
                } else {
                    tree!(IR::NodeByLabelScan(relationship.from.clone()))
                };
                tree!(IR::ExpandInto(relationship.clone()), scan)
            } else if self.visited.contains(&relationship.from.alias.id)
                && self.visited.contains(&relationship.to.alias.id)
            {
                tree!(IR::ExpandInto(relationship.clone()))
            } else if relationship.min_hops.is_some() {
                tree!(IR::CondVarLenTraverse(relationship.clone()))
            } else {
                tree!(IR::CondTraverse(relationship.clone()))
            };
            self.visited.insert(relationship.from.alias.id);
            self.visited.insert(relationship.to.alias.id);
            self.visited.insert(relationship.alias.id);
            for relationship in iter {
                res = if relationship.from.alias.id == relationship.to.alias.id {
                    let scan = if relationship.from.labels.is_empty() {
                        tree!(IR::AllNodeScan(relationship.from.clone()))
                    } else {
                        tree!(IR::NodeByLabelScan(relationship.from.clone()))
                    };
                    tree!(IR::ExpandInto(relationship.clone()), scan, res)
                } else if self.visited.contains(&relationship.from.alias.id)
                    && self.visited.contains(&relationship.to.alias.id)
                {
                    tree!(IR::ExpandInto(relationship.clone()), res)
                } else if relationship.min_hops.is_some() {
                    tree!(IR::CondVarLenTraverse(relationship.clone()), res)
                } else {
                    tree!(IR::CondTraverse(relationship.clone()), res)
                };
                self.visited.insert(relationship.from.alias.id);
                self.visited.insert(relationship.to.alias.id);
                self.visited.insert(relationship.alias.id);
            }
            let paths = component.paths();
            if !paths.is_empty() {
                res = tree!(IR::PathBuilder(paths), res);
            }
            vec.push(res);
        }
        let mut res = if vec.len() == 1 {
            vec.pop().unwrap()
        } else {
            tree!(IR::CartesianProduct; vec)
        };
        if let Some(filter) = filter {
            let mut extractable = vec![];
            let mut inline = vec![];
            let visited_max = self.visited.iter().copied().max().unwrap_or(0);
            let filter_max = Self::max_var_id_in_expr(&filter);
            self.next_var_id = self.next_var_id.max(visited_max + 1).max(filter_max + 1);
            let rebuilt = Self::collect_patterns_and_rebuild(
                filter.root(),
                &mut extractable,
                &mut inline,
                &mut self.next_var_id,
                true,
            );

            // When there are inline patterns, recursively decompose the
            // rebuilt expression into multiplexer / semi-apply / filter nodes.
            if !inline.is_empty() {
                let inline_map: HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>> =
                    inline.into_iter().map(|(v, g)| (v.id, g)).collect();
                res = self.expr_to_plan(rebuilt.root(), &inline_map, res);
            } else if !matches!(rebuilt.root().data(), ExprIR::Bool(true)) {
                res = tree!(IR::Filter(Arc::new(rebuilt)), res);
            }

            // Apply SemiApply/AntiSemiApply for each extractable pattern
            for (graph, is_anti) in extractable {
                let saved = self.visited.clone();
                let mut sub_plan = self.plan_match(&graph, None);
                self.visited = saved;
                Self::add_argument_to_leaves(&mut sub_plan);
                if is_anti {
                    res = tree!(IR::AntiSemiApply, res, sub_plan);
                } else {
                    res = tree!(IR::SemiApply, res, sub_plan);
                }
            }
        }
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_project(
        &mut self,
        exprs: Vec<(Variable, QueryExpr<Variable>)>,
        copy_from_parent: Vec<(Variable, Variable)>,
        orderby: Vec<(QueryExpr<Variable>, bool)>,
        skip: Option<QueryExpr<Variable>>,
        limit: Option<QueryExpr<Variable>>,
        filter: Option<QueryExpr<Variable>>,
        distinct: bool,
        write: bool,
    ) -> DynTree<IR> {
        // Clear visited set for the new scope — after WITH/RETURN, only the
        // projected (and copied) variables are in scope.  Previous-scope IDs
        // must not leak into pattern-predicate sub-plans.
        self.visited.clear();
        for expr in &exprs {
            self.visited.insert(expr.0.id);
        }
        for (new_var, _) in &copy_from_parent {
            self.visited.insert(new_var.id);
        }
        let mut res = if exprs.iter().any(|e| e.1.is_aggregation()) {
            let mut group_by_keys = Vec::new();
            let mut aggregations = Vec::new();
            let mut names = Vec::new();
            for (name, expr) in exprs {
                names.push(name.clone());
                if expr.is_aggregation() {
                    aggregations.push((name, expr));
                } else {
                    group_by_keys.push((name, expr));
                }
            }
            tree!(IR::Aggregate(names, group_by_keys, aggregations))
        } else {
            tree!(IR::Project(exprs, copy_from_parent))
        };
        if write {
            res.root_mut().push_child(IR::Commit);
        }
        if distinct {
            res = tree!(IR::Distinct, res);
        }
        if !orderby.is_empty() {
            res = tree!(IR::Sort(orderby), res);
        }
        if let Some(skip_expr) = skip {
            res = tree!(IR::Skip(skip_expr), res);
        }
        if let Some(limit_expr) = limit {
            res = tree!(IR::Limit(limit_expr), res);
        }
        if let Some(filter) = filter {
            let mut extractable = vec![];
            let mut inline = vec![];
            let visited_max = self.visited.iter().copied().max().unwrap_or(0);
            let filter_max = Self::max_var_id_in_expr(&filter);
            self.next_var_id = self.next_var_id.max(visited_max + 1).max(filter_max + 1);
            let rebuilt = Self::collect_patterns_and_rebuild(
                filter.root(),
                &mut extractable,
                &mut inline,
                &mut self.next_var_id,
                true,
            );

            if !matches!(rebuilt.root().data(), ExprIR::Bool(true)) {
                if inline.is_empty() {
                    res = tree!(IR::Filter(Arc::new(rebuilt)), res);
                } else {
                    let inline_map: HashMap<u32, QueryGraph<Arc<String>, Arc<String>, Variable>> =
                        inline.into_iter().map(|(v, g)| (v.id, g)).collect();
                    res = self.expr_to_plan(rebuilt.root(), &inline_map, res);
                }
            }

            for (graph, is_anti) in extractable {
                let saved = self.visited.clone();
                let mut sub_plan = self.plan_match(&graph, None);
                self.visited = saved;
                Self::add_argument_to_leaves(&mut sub_plan);
                if is_anti {
                    res = tree!(IR::AntiSemiApply, res, sub_plan);
                } else {
                    res = tree!(IR::SemiApply, res, sub_plan);
                }
            }
        }
        res
    }

    fn plan_query(
        &mut self,
        q: Vec<QueryIR<Variable>>,
        write: bool,
    ) -> DynTree<IR> {
        let mut plans = Vec::with_capacity(q.len());
        for ir in q {
            plans.push(self.plan(ir));
        }
        let mut iter = plans.into_iter().rev();
        let mut res = iter.next().unwrap();
        let mut idx = res.root().idx();
        while matches!(res.node(idx).data(), |IR::Sort(_)| IR::Skip(_)
            | IR::Limit(_)
            | IR::Distinct
            | IR::Filter(_)
            | IR::SemiApply
            | IR::AntiSemiApply
            | IR::OrApplyMultiplexer(_))
        {
            idx = res.node(idx).child(0).idx();
        }
        if matches!(res.node(idx).data(), |IR::Project(_, _)| IR::Aggregate(
            _,
            _,
            _
        )) && res.node(idx).num_children() > 0
            && matches!(res.node(idx).child(0).data(), IR::Commit)
        {
            idx = res.node(idx).child(0).idx();
        }
        for n in iter {
            if res.node(idx).num_children() > 0 {
                idx = res
                    .node_mut(idx)
                    .child_mut(0)
                    .push_sibling_tree(Side::Left, n);
            } else {
                idx = res.node_mut(idx).push_child_tree(n);
            }
            while matches!(res.node(idx).data(), |IR::Sort(_)| IR::Skip(_)
                | IR::Limit(_)
                | IR::Distinct
                | IR::Filter(_)
                | IR::SemiApply
                | IR::AntiSemiApply
                | IR::OrApplyMultiplexer(_))
            {
                idx = res.node(idx).child(0).idx();
            }
            if matches!(res.node(idx).data(), |IR::Project(_, _)| IR::Aggregate(
                _,
                _,
                _
            )) && res.node(idx).num_children() > 0
                && matches!(res.node(idx).child(0).data(), IR::Commit)
            {
                idx = res.node(idx).child(0).idx();
            }
        }
        if write {
            res = tree!(IR::Commit, res);
        }
        res
    }

    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn plan(
        &mut self,
        ir: BoundQueryIR,
    ) -> DynTree<IR> {
        match ir {
            QueryIR::Call(proc, exprs, named_outputs, filter) => {
                if let Some(filter) = filter {
                    return tree!(
                        IR::Filter(filter),
                        tree!(IR::ProcedureCall(proc, exprs, named_outputs))
                    );
                }
                if proc.name == "db.idx.fulltext.drop" {
                    let ExprIR::String(label) = exprs[0].root().data() else {
                        unreachable!()
                    };
                    return tree!(IR::DropIndex {
                        label: label.clone(),
                        attrs: vec![],
                        index_type: IndexType::Fulltext,
                        entity_type: EntityType::Node,
                    });
                }
                if proc.name == "db.idx.fulltext.createNodeIndex" {
                    let label = match exprs[0].root().data() {
                        ExprIR::String(label) => label.clone(),
                        ExprIR::Map => {
                            let mut ret = None;
                            for child in exprs[0].root().children() {
                                if let ExprIR::String(label) = child.data()
                                    && label.as_str() == "label"
                                {
                                    ret = Some(label.clone());
                                    break;
                                }
                            }
                            ret.unwrap_or_else(|| {
                                unreachable!();
                            })
                        }
                        _ => unreachable!(),
                    };
                    return tree!(IR::CreateIndex {
                        label,
                        attrs: vec![],
                        index_type: IndexType::Fulltext,
                        entity_type: EntityType::Node,
                        options: None,
                    });
                }
                tree!(IR::ProcedureCall(proc, exprs, named_outputs))
            }
            QueryIR::Match {
                pattern,
                filter,
                optional,
            } => {
                if optional {
                    // Compute optional variables BEFORE plan_match adds them to visited
                    let optional_vars: Vec<Variable> = pattern
                        .variables()
                        .iter()
                        .filter(|v| !self.visited.contains(&v.id))
                        .cloned()
                        .collect();
                    let all_visited = pattern
                        .variables()
                        .iter()
                        .all(|v| self.visited.contains(&v.id));
                    let mut match_plan = self.plan_match(&pattern, filter);
                    Self::add_argument_to_leaves(&mut match_plan);
                    if all_visited {
                        tree!(IR::Apply, tree!(IR::Optional(optional_vars), match_plan))
                    } else {
                        tree!(IR::Optional(optional_vars), match_plan)
                    }
                } else {
                    self.plan_match(&pattern, filter)
                }
            }
            QueryIR::Unwind(expr, alias) => tree!(IR::Unwind(expr, alias)),
            QueryIR::Merge(pattern, on_create_set_items, on_match_set_items) => {
                let create_pattern = pattern.filter_visited(&self.visited);
                let mut match_branch = self.plan_match(&pattern, None);
                Self::add_argument_to_leaves(&mut match_branch);

                tree!(
                    IR::Merge(create_pattern, on_create_set_items, on_match_set_items),
                    match_branch
                )
            }
            QueryIR::Create(pattern) => {
                tree!(IR::Create(pattern.filter_visited(&self.visited)))
            }
            QueryIR::Delete(exprs, is_detach) => tree!(IR::Delete(exprs, is_detach)),
            QueryIR::Set(items) => tree!(IR::Set(items)),
            QueryIR::Remove(items) => tree!(IR::Remove(items)),
            QueryIR::LoadCsv {
                file_path,
                headers,
                delimiter,
                var,
            } => {
                tree!(IR::LoadCsv {
                    file_path,
                    headers,
                    delimiter,
                    var,
                })
            }
            QueryIR::With {
                distinct,
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                write,
                ..
            } => self.plan_project(
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                filter,
                distinct,
                write,
            ),
            QueryIR::Return {
                distinct,
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                write,
                ..
            } => self.plan_project(
                exprs,
                copy_from_parent,
                orderby,
                skip,
                limit,
                None,
                distinct,
                write,
            ),
            QueryIR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options,
            } => tree!(IR::CreateIndex {
                label,
                attrs,
                index_type,
                entity_type,
                options
            }),
            QueryIR::DropIndex {
                label,
                attrs,
                index_type,
                entity_type,
            } => {
                tree!(IR::DropIndex {
                    label,
                    attrs,
                    index_type,
                    entity_type
                })
            }
            QueryIR::Query(q, write) => self.plan_query(q, write),
        }
    }
}
