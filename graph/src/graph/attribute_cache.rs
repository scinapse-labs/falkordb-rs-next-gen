//! In-memory concurrent cache for entity attributes.
//!
//! Provides a shared, version-stamped, entity-level cache that sits between
//! [`super::attribute_store::AttributeStore`] and fjall.  Writes go to the
//! cache first (marked dirty); fjall is updated asynchronously when the
//! memory threshold is exceeded.
//!
//! The cache is shared across MVCC versions via `Arc` so that
//! `AttributeStore::new_version()` costs only a pointer increment.
//!
//! Uses [`quick_cache`] internally — a sharded, lock-free concurrent cache
//! with CLOCK-based approximate LRU eviction and byte-weighted capacity.

use quick_cache::Weighter;
use quick_cache::sync::Cache;

use crate::runtime::value::Value;

/// Per-entity cached attributes.
#[derive(Clone)]
struct CachedEntity {
    /// Sorted by `attr_idx` for O(log n) binary-search lookups.
    attrs: Vec<(u16, Value)>,
    /// Graph version when this entry was written/populated.
    version: u64,
    /// `true` when the entry has not yet been flushed to fjall.
    dirty: bool,
}

/// Weighter that estimates the heap footprint of each cached entity.
#[derive(Clone)]
struct EntityWeighter;

impl Weighter<u64, CachedEntity> for EntityWeighter {
    fn weight(
        &self,
        _key: &u64,
        val: &CachedEntity,
    ) -> u64 {
        let base = val.attrs.len() * (std::mem::size_of::<u16>() + std::mem::size_of::<Value>());
        let heap: usize = val.attrs.iter().map(|(_, v)| v.heap_size()).sum();
        // Minimum weight of 1 to satisfy quick_cache invariant.
        (base + heap + std::mem::size_of::<CachedEntity>()).max(1) as u64
    }
}

/// Shared, version-stamped, entity-level attribute cache.
///
/// Thread-safety is handled internally by `quick_cache` via sharded locks —
/// concurrent reads do not block each other.
pub struct AttributeCache {
    entries: Cache<u64, CachedEntity, EntityWeighter>,
}

impl AttributeCache {
    /// Create a new cache with the given byte budget.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        // Use a modest item estimate — quick_cache grows internally as needed.
        // The weight_capacity is the actual byte budget that governs eviction.
        Self {
            entries: Cache::with_weighter(1024, max_bytes as u64, EntityWeighter),
        }
    }

    /// Look up a single attribute for an entity by index.
    ///
    /// Returns `Some(Some(value))` on cache hit with the attribute present,
    /// `Some(None)` on cache hit but attribute absent, and `None` on cache
    /// miss.
    #[must_use]
    pub fn get_attr(
        &self,
        entity_id: u64,
        attr_idx: u16,
        version: u64,
    ) -> Option<Option<Value>> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None; // newer uncommitted data — ignore
        }
        Some(
            entry
                .attrs
                .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
                .ok()
                .map(|pos| entry.attrs[pos].1.clone()),
        )
    }

    /// Return all cached attributes for an entity.
    ///
    /// Returns `None` on cache miss or version mismatch.
    #[must_use]
    pub fn get_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<Vec<(u16, Value)>> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some(entry.attrs)
    }

    /// Return all cached attributes for an entity along with the dirty flag.
    ///
    /// Returns `None` on cache miss or version mismatch.
    #[must_use]
    pub fn get_entity_with_dirty(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<(Vec<(u16, Value)>, bool)> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some((entry.attrs, entry.dirty))
    }

    /// Check whether an entity has *any* cached attributes.
    #[must_use]
    pub fn has_entity(
        &self,
        entity_id: u64,
        version: u64,
    ) -> Option<bool> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some(!entry.attrs.is_empty())
    }

    /// Insert (or replace) the full attribute set for an entity.
    ///
    /// The incoming `attrs` are sorted by `attr_idx` before storing to maintain
    /// the invariant required by binary searches in `get_attr`, `contains_attr`,
    /// and other methods that rely on sorted access.
    pub fn insert_entity(
        &self,
        entity_id: u64,
        mut attrs: Vec<(u16, Value)>,
        version: u64,
        dirty: bool,
    ) {
        // Ensure attrs are sorted by attr_idx to support binary searches.
        attrs.sort_by_key(|item| item.0);
        let entry = CachedEntity {
            attrs,
            version,
            dirty,
        };
        self.entries.insert(entity_id, entry);
    }

    /// Insert or update a cache entry only if not overwriting a newer or dirty entry.
    ///
    /// This is used by `populate_cache_from_fjall` to safely cache fjall reads without
    /// overwriting in-flight dirty writes. The insert only proceeds if:
    /// - No cache entry exists for this entity, OR
    /// - The existing entry is from an older version AND not dirty
    ///
    /// Returns `true` if the insert proceeded, `false` if it was skipped due to
    /// a newer/dirty entry already in cache.
    ///
    /// **Note:** There is a narrow TOCTOU window between the `get` check and the
    /// subsequent `insert_entity` call.  A concurrent writer could insert a dirty
    /// entry in that gap, which this clean insert would then overwrite.  In
    /// practice this is benign because (a) the window is sub-microsecond,
    /// (b) writers are serialized so only one write transaction is active, and
    /// (c) readers always hold an older MVCC version so the version check
    /// (`>=`) catches the common case.
    #[must_use]
    pub fn insert_entity_if_older(
        &self,
        entity_id: u64,
        attrs: Vec<(u16, Value)>,
        version: u64,
    ) -> bool {
        // Check if a newer or dirty entry already exists
        if let Some(existing) = self.entries.get(&entity_id) {
            // Skip if existing entry is newer or dirty (in-flight write)
            if existing.version >= version || existing.dirty {
                return false;
            }
        }
        // Safe to insert or update with this fjall read
        self.insert_entity(entity_id, attrs, version, false);
        true
    }

    /// Remove a single entity from the cache.
    pub fn invalidate(
        &self,
        entity_id: u64,
    ) {
        self.entries.remove(&entity_id);
    }

    /// Batch-invalidate entities (used during rollback).
    pub fn invalidate_batch(
        &self,
        entity_ids: &roaring::RoaringTreemap,
    ) {
        for id in entity_ids {
            self.entries.remove(&id);
        }
    }

    /// Check whether an attr already exists for an entity in the cache.
    ///
    /// Returns `Some(true/false)` on cache hit, `None` on miss.
    #[must_use]
    pub fn contains_attr(
        &self,
        entity_id: u64,
        attr_idx: u16,
        version: u64,
    ) -> Option<bool> {
        let entry = self.entries.get(&entity_id)?;
        if entry.version > version {
            return None;
        }
        Some(
            entry
                .attrs
                .binary_search_by_key(&attr_idx, |(idx, _)| *idx)
                .is_ok(),
        )
    }

    /// Remove a single attribute from a cached entity, keeping the entry dirty.
    ///
    /// This is used by `remove_attr` to remove only a specific attribute while
    /// preserving other cached attributes. The entry remains marked dirty and
    /// will be flushed to fjall along with other pending changes.
    ///
    /// Returns `true` if the attribute was found and removed, `false` if the
    /// entity was not in cache or the attribute didn't exist.
    #[must_use]
    pub fn remove_attr_from_entity(
        &self,
        entity_id: u64,
        attr_idx: u16,
    ) -> bool {
        if let Some(mut entry) = self.entries.get(&entity_id)
            && let Ok(pos) = entry.attrs.binary_search_by_key(&attr_idx, |(idx, _)| *idx)
        {
            entry.attrs.remove(pos);
            entry.dirty = true;
            // Update the cache with the modified entry
            self.entries.insert(entity_id, entry);
            return true;
        }
        false
    }

    /// Returns `true` when the cache is over its memory budget.
    #[must_use]
    pub fn over_budget(&self) -> bool {
        self.entries.weight() > self.entries.capacity()
    }

    /// Collect up to `n` dirty entries for flushing.
    ///
    /// Collected entries are **removed** from the cache.  The caller is
    /// responsible for writing them to fjall.
    #[must_use]
    pub fn collect_dirty_lru(
        &self,
        n: usize,
    ) -> Vec<(u64, Vec<(u16, Value)>)> {
        let mut result = Vec::with_capacity(n);
        // Iterate and collect dirty entries.
        for (entity_id, entry) in self.entries.iter() {
            if result.len() >= n {
                break;
            }
            if entry.dirty {
                result.push((entity_id, entry.attrs.clone()));
            }
        }
        // Remove collected entries from cache.
        for (id, _) in &result {
            self.entries.remove(id);
        }
        result
    }

    /// Total estimated bytes currently in the cache.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.entries.weight() as usize
    }
}
