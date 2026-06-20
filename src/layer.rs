//! A global write-quota [`Layer`] for OpenDAL.
//!
//! `QuotaLayer` behaves like a "bytes remaining on disk" check: every byte
//! that flows through a `write()` call is counted against a configured
//! limit, and once that limit is reached, further writes are rejected with
//! an error. Reads, lists, stats, deletes, etc. are all left untouched.
//!
//! Usage is the same shape as opendal's other layers like `ThrottleLayer` or
//! `RouteLayer`: construct it and hand it to `Operator::layer`.
//!
//! ```ignore
//! // Adjust the `crate::quota_layer` path below to wherever you place this
//! // module in your own crate (e.g. `use my_crate::quota_layer::...`).
//! use std::collections::HashMap;
//! use std::sync::Mutex;
//!
//! use async_trait::async_trait;
//! use opendal::{Operator, Result, services};
//!
//! use crate::quota_layer::{QuotaLayer, QuotaTracker};
//!
//! /// Toy in-memory tracker. In real usage this would read/write a row in
//! /// whatever database backs your quota accounting.
//! #[derive(Default)]
//! struct InMemoryTracker(Mutex<HashMap<String, u64>>);
//!
//! #[async_trait]
//! impl QuotaTracker for InMemoryTracker {
//!     async fn get_bytes_written(&self, id: &str) -> Result<u64> {
//!         Ok(*self.0.lock().unwrap().get(id).unwrap_or(&0))
//!     }
//!
//!     async fn set_bytes_written(&self, id: &str, bytes: u64) -> Result<()> {
//!         self.0.lock().unwrap().insert(id.to_string(), bytes);
//!         Ok(())
//!     }
//! }
//!
//! # async fn run() -> Result<()> {
//! let op = Operator::new(services::Memory::default())?
//!     .layer(QuotaLayer::new("tenant-a", InMemoryTracker::default(), 1024))
//!     .finish();
//!
//! op.write("foo.txt", "hello world").await?;
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex as AsyncMutex;

use opendal::raw::*;
use opendal::{Buffer, Error, ErrorKind, Metadata, Result};

/// Persistence for how many bytes have been written under a given quota id.
///
/// Implement this against your database (or cache, or whatever else backs
/// your quota accounting). `QuotaLayer` calls `get_bytes_written` at most
/// once per process per id (to warm an in-memory cache) and calls
/// `set_bytes_written` every time that cached total changes.
///
/// # Concurrency note
///
/// `QuotaLayer` serializes its own check-then-update sequence with an
/// internal lock, so multiple writers sharing one `QuotaLayer` instance
/// (and therefore one process) can't race each other. It can NOT make a
/// plain get/set pair atomic across multiple processes sharing the same
/// `id`: two processes could both read the same starting total before
/// either calls `set_bytes_written`, and the loser's update would clobber
/// the winner's. If you need correctness across processes, implement
/// `set_bytes_written` as a conditional/atomic update in your store (e.g. an
/// `UPDATE ... SET bytes = ? WHERE id = ? AND bytes = ?` or a native atomic
/// increment) rather than a blind overwrite.
#[async_trait]
pub trait QuotaTracker: Send + Sync + 'static {
    /// Return the total number of bytes written so far for `id`. An id that
    /// has never been seen before should be treated as `0`.
    async fn get_bytes_written(&self, id: &str) -> Result<u64>;

    /// Persist `bytes` as the new total number of bytes written for `id`.
    async fn set_bytes_written(&self, id: &str, bytes: u64) -> Result<()>;
}

/// Shared state behind one `QuotaLayer`, cloned (via `Arc`) into every
/// accessor and writer it produces.
struct QuotaState<T: QuotaTracker> {
    id: String,
    tracker: T,
    limit: u64,
    /// In-process cache of the last known total. `None` until the first
    /// write touches it, at which point it's warmed from the tracker.
    cache: AsyncMutex<Option<u64>>,
}

impl<T: QuotaTracker> QuotaState<T> {
    /// Try to reserve `len` additional bytes against the quota, persisting
    /// the new total through the tracker. Returns an error - and reserves
    /// nothing - if doing so would exceed the configured limit.
    async fn reserve(&self, len: u64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        let mut cache = self.cache.lock().await;
        let current = match *cache {
            Some(v) => v,
            None => self.tracker.get_bytes_written(&self.id).await?,
        };

        let new_total = current.saturating_add(len);
        if new_total > self.limit {
            // Make sure the cache reflects what we just confirmed, even
            // though this particular reservation is being rejected.
            *cache = Some(current);
            return Err(Error::new(
                ErrorKind::RateLimited,
                format!(
                    "write quota exceeded for '{}': {} bytes used, {} byte write requested, {} byte limit",
                    self.id, current, len, self.limit
                ),
            )
                .with_context("quota_id", self.id.clone())
                .with_context("quota_limit", self.limit.to_string())
                .with_context("quota_used", current.to_string())
                .with_context("quota_requested", len.to_string()));
        }

        self.tracker.set_bytes_written(&self.id, new_total).await?;
        *cache = Some(new_total);
        Ok(())
    }

    /// Best-effort release of `len` previously reserved bytes that turned
    /// out not to be durably written (e.g. because the underlying write
    /// failed or was aborted).
    async fn release(&self, len: u64) {
        if len == 0 {
            return;
        }

        let mut cache = self.cache.lock().await;
        let current = cache.unwrap_or(0);
        let new_total = current.saturating_sub(len);
        // Best effort: if the store is unreachable there's nothing more
        // useful to do here than try; the next successful get/set will
        // resync the cache.
        let _ = self.tracker.set_bytes_written(&self.id, new_total).await;
        *cache = Some(new_total);
    }
}

/// A global write-quota layer.
///
/// Caps total bytes written across an operator at a configured limit,
/// persisting usage via a [`QuotaTracker`] so it survives restarts. Reads
/// are never affected.
///
/// # Example
///
/// ```ignore
/// # use opendal::{Operator, Result, services};
/// # use crate::quota_layer::QuotaLayer;
/// # async fn run(tracker: impl crate::quota_layer::QuotaTracker) -> Result<()> {
/// let op = Operator::new(services::Memory::default())?
///     .layer(QuotaLayer::new("tenant-a", tracker, 10 * 1024 * 1024))
///     .finish();
/// # Ok(())
/// # }
/// ```
pub struct QuotaLayer<T: QuotaTracker> {
    state: Arc<QuotaState<T>>,
}

impl<T: QuotaTracker> Clone for QuotaLayer<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<T: QuotaTracker> QuotaLayer<T> {
    /// Create a new `QuotaLayer`.
    ///
    /// - `id`: identifies which quota "bucket" this operator enforces
    ///   against. Passed through to every call to `tracker`'s get/set
    ///   methods, so one `QuotaTracker` implementation can back many
    ///   independently-tracked operators (e.g. one per tenant).
    /// - `tracker`: the get/set persistence backing this quota's usage.
    /// - `limit_bytes`: the total number of bytes this id is allowed to
    ///   have written, ever (cumulative, like free space on a disk).
    pub fn new(id: impl Into<String>, tracker: T, limit_bytes: u64) -> Self {
        Self {
            state: Arc::new(QuotaState {
                id: id.into(),
                tracker,
                limit: limit_bytes,
                cache: AsyncMutex::new(None),
            }),
        }
    }
}

impl<A: Access, T: QuotaTracker> Layer<A> for QuotaLayer<T> {
    type LayeredAccess = QuotaAccessor<A, T>;

    fn layer(&self, inner: A) -> Self::LayeredAccess {
        QuotaAccessor {
            inner,
            state: self.state.clone(),
        }
    }
}

pub struct QuotaAccessor<A: Access, T: QuotaTracker> {
    inner: A,
    state: Arc<QuotaState<T>>,
}

// Manual `Debug` so we don't need to require `T: Debug` (database clients
// generally don't implement it, and there's nothing useful to print there
// anyway). `LayeredAccess` requires `Debug`.
impl<A: Access, T: QuotaTracker> fmt::Debug for QuotaAccessor<A, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuotaAccessor")
            .field("id", &self.state.id)
            .field("limit", &self.state.limit)
            .finish_non_exhaustive()
    }
}

impl<A: Access, T: QuotaTracker> LayeredAccess for QuotaAccessor<A, T> {
    type Inner = A;
    type Reader = A::Reader;
    type Writer = QuotaWriter<A::Writer, T>;
    type Lister = A::Lister;
    type Deleter = A::Deleter;
    type Copier = A::Copier;

    fn inner(&self) -> &Self::Inner {
        &self.inner
    }

    // Reads are untouched - this is a write-only quota.
    async fn read(&self, path: &str, args: OpRead) -> Result<(RpRead, Self::Reader)> {
        self.inner.read(path, args).await
    }

    async fn write(&self, path: &str, args: OpWrite) -> Result<(RpWrite, Self::Writer)> {
        let (rp, w) = self.inner.write(path, args).await?;
        Ok((rp, QuotaWriter::new(w, self.state.clone())))
    }

    async fn delete(&self) -> Result<(RpDelete, Self::Deleter)> {
        self.inner.delete().await
    }

    async fn list(&self, path: &str, args: OpList) -> Result<(RpList, Self::Lister)> {
        self.inner.list(path, args).await
    }
}

/// Wraps an inner writer, charging every chunk it writes against the quota
/// before letting it through.
pub struct QuotaWriter<W, T: QuotaTracker> {
    inner: W,
    state: Arc<QuotaState<T>>,
    /// Bytes reserved against the quota during this writer's lifetime that
    /// haven't been confirmed durable yet (i.e. not yet `close()`d). Used to
    /// roll the reservation back on `abort()`.
    reserved: u64,
}

impl<W, T: QuotaTracker> QuotaWriter<W, T> {
    fn new(inner: W, state: Arc<QuotaState<T>>) -> Self {
        Self {
            inner,
            state,
            reserved: 0,
        }
    }
}

impl<W: oio::Write, T: QuotaTracker> oio::Write for QuotaWriter<W, T> {
    async fn write(&mut self, bs: Buffer) -> Result<()> {
        let len = bs.len() as u64;
        self.state.reserve(len).await?;
        self.reserved += len;

        if let Err(e) = self.inner.write(bs).await {
            // The bytes never made it to the backend - give the quota back.
            self.reserved -= len;
            self.state.release(len).await;
            return Err(e);
        }

        Ok(())
    }

    async fn close(&mut self) -> Result<Metadata> {
        let meta = self.inner.close().await?;
        // Everything we reserved made it through and is now finalized.
        self.reserved = 0;
        Ok(meta)
    }

    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await?;
        let to_release = self.reserved;
        self.reserved = 0;
        self.state.release(to_release).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use opendal::{Operator, services};
    use tokio::sync::Mutex;

    use super::*;

    /// Simple in-memory stand-in for a DB-backed tracker, for tests.
    #[derive(Default)]
    struct MemoryTracker(Mutex<HashMap<String, u64>>);

    #[async_trait]
    impl QuotaTracker for MemoryTracker {
        async fn get_bytes_written(&self, id: &str) -> Result<u64> {
            Ok(*self.0.lock().await.get(id).unwrap_or(&0))
        }

        async fn set_bytes_written(&self, id: &str, bytes: u64) -> Result<()> {
            self.0.lock().await.insert(id.to_string(), bytes);
            Ok(())
        }
    }

    fn build_op(id: &str, tracker: Arc<MemoryTracker>, limit: u64) -> Operator {
        Operator::new(services::Memory::default())
            .unwrap()
            .layer(QuotaLayer::new(id, SharedTracker(tracker), limit))
            .finish()
    }

    /// `QuotaTracker` needs an owned type per layer; this thin wrapper lets
    /// tests share one `MemoryTracker` across multiple operators/layers so
    /// they can assert on persisted totals afterward.
    struct SharedTracker(Arc<MemoryTracker>);

    #[async_trait]
    impl QuotaTracker for SharedTracker {
        async fn get_bytes_written(&self, id: &str) -> Result<u64> {
            self.0.get_bytes_written(id).await
        }

        async fn set_bytes_written(&self, id: &str, bytes: u64) -> Result<()> {
            self.0.set_bytes_written(id, bytes).await
        }
    }

    #[tokio::test]
    async fn writes_within_quota_succeed_and_are_tracked() {
        let tracker = Arc::new(MemoryTracker::default());
        let op = build_op("tenant-a", tracker.clone(), 1024);

        op.write("a.txt", "hello world").await.unwrap();
        assert_eq!(
            tracker.get_bytes_written("tenant-a").await.unwrap(),
            "hello world".len() as u64
        );

        op.write("b.txt", "more data here").await.unwrap();
        assert_eq!(
            tracker.get_bytes_written("tenant-a").await.unwrap(),
            ("hello world".len() + "more data here".len()) as u64
        );
    }

    #[tokio::test]
    async fn write_exceeding_quota_is_rejected() {
        let tracker = Arc::new(MemoryTracker::default());
        let op = build_op("tenant-b", tracker.clone(), 10);

        let err = op
            .write("too-big.txt", "this is way more than 10 bytes")
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::RateLimited);

        // Nothing should have been charged for the rejected write.
        assert_eq!(tracker.get_bytes_written("tenant-b").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn quota_persists_across_separate_operators_with_same_id() {
        let tracker = Arc::new(MemoryTracker::default());

        let op1 = build_op("tenant-c", tracker.clone(), 20);
        op1.write("a.txt", "0123456789").await.unwrap(); // 10 bytes

        // A brand new operator instance, same id, same backing tracker -
        // simulates a process restart that re-warms from the DB.
        let op2 = build_op("tenant-c", tracker.clone(), 20);
        op2.write("b.txt", "0123456789").await.unwrap(); // another 10 bytes, at the limit

        let err = op2.write("c.txt", "x").await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::RateLimited);

        assert_eq!(tracker.get_bytes_written("tenant-c").await.unwrap(), 20);
    }

    #[tokio::test]
    async fn aborted_write_releases_its_reservation() {
        let tracker = Arc::new(MemoryTracker::default());
        let op = build_op("tenant-f", tracker.clone(), 10);

        // Stream a writer manually so we can abort it ourselves instead of
        // closing it.
        let mut w = op.writer("a.txt").await.unwrap();
        w.write("12345").await.unwrap(); // 5/10 bytes reserved
        assert_eq!(tracker.get_bytes_written("tenant-f").await.unwrap(), 5);

        w.abort().await.unwrap();
        // The reservation should have been given back since the data was
        // never actually finalized.
        assert_eq!(tracker.get_bytes_written("tenant-f").await.unwrap(), 0);

        // Confirm the full limit is usable again afterwards.
        op.write("b.txt", "0123456789").await.unwrap();
        assert_eq!(tracker.get_bytes_written("tenant-f").await.unwrap(), 10);
    }

    #[tokio::test]
    async fn different_ids_have_independent_quotas() {
        let tracker = Arc::new(MemoryTracker::default());
        let op_a = build_op("tenant-d", tracker.clone(), 5);
        let op_e = build_op("tenant-e", tracker.clone(), 5);

        op_a.write("a.txt", "12345").await.unwrap();
        assert!(op_a.write("a2.txt", "x").await.is_err());

        // tenant-e's quota is untouched by tenant-d's usage.
        op_e.write("e.txt", "12345").await.unwrap();
    }
}
