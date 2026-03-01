/// Entity type that can be indexed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityType {
    /// Index on node properties
    Node,
    /// Index on relationship properties
    Relationship,
}
