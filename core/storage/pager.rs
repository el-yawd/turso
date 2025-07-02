use super::btree::BTreePage;
use super::page_cache::{CacheError, CacheResizeResult, DumbLruPageCache, PageCacheKey};
use super::sqlite3_ondisk::{begin_write_btree_page, DATABASE_HEADER_SIZE};
use super::wal::{CheckpointMode, CheckpointStatus};
use crate::result::LimboResult;
use crate::storage::btree::BTreePageInner;
use crate::storage::buffer_pool::BufferPool;
use crate::storage::database::DatabaseStorage;
use crate::storage::header_accessor;
use crate::storage::sqlite3_ondisk::{self, DatabaseHeader, PageContent, PageType};
use crate::storage::wal::{CheckpointResult, Wal, WalFsyncStatus};
use crate::types::CursorResult;
use crate::{Buffer, LimboError, Result};
use crate::{Completion, WalFile};
use parking_lot::RwLock;
use std::cell::{OnceCell, RefCell, UnsafeCell};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{trace, Level};

pub struct PageInner {
    pub flags: AtomicUsize,
    pub contents: Option<PageContent>,
    pub id: usize,
}

#[derive(Debug)]
pub struct Page {
    pub inner: UnsafeCell<PageInner>,
}

// Concurrency control of pages will be handled by the pager, we won't wrap Page with RwLock
// because that is bad bad.
pub type PageRef = Arc<Page>;

/// Page is up-to-date.
const PAGE_UPTODATE: usize = 0b001;
/// Page is locked for I/O to prevent concurrent access.
const PAGE_LOCKED: usize = 0b010;
/// Page had an I/O error.
const PAGE_ERROR: usize = 0b100;
/// Page is dirty. Flush needed.
const PAGE_DIRTY: usize = 0b1000;
/// Page's contents are loaded in memory.
const PAGE_LOADED: usize = 0b10000;

impl Page {
    pub fn new(id: usize) -> Self {
        Self {
            inner: UnsafeCell::new(PageInner {
                flags: AtomicUsize::new(0),
                contents: None,
                id,
            }),
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn get(&self) -> &mut PageInner {
        unsafe { &mut *self.inner.get() }
    }

    pub fn get_contents(&self) -> &mut PageContent {
        self.get().contents.as_mut().unwrap()
    }

    pub fn is_uptodate(&self) -> bool {
        self.get().flags.load(Ordering::SeqCst) & PAGE_UPTODATE != 0
    }

    pub fn set_uptodate(&self) {
        self.get().flags.fetch_or(PAGE_UPTODATE, Ordering::SeqCst);
    }

    pub fn clear_uptodate(&self) {
        self.get().flags.fetch_and(!PAGE_UPTODATE, Ordering::SeqCst);
    }

    pub fn is_locked(&self) -> bool {
        self.get().flags.load(Ordering::SeqCst) & PAGE_LOCKED != 0
    }

    pub fn set_locked(&self) {
        self.get().flags.fetch_or(PAGE_LOCKED, Ordering::SeqCst);
    }

    pub fn clear_locked(&self) {
        self.get().flags.fetch_and(!PAGE_LOCKED, Ordering::SeqCst);
    }

    pub fn is_error(&self) -> bool {
        self.get().flags.load(Ordering::SeqCst) & PAGE_ERROR != 0
    }

    pub fn set_error(&self) {
        self.get().flags.fetch_or(PAGE_ERROR, Ordering::SeqCst);
    }

    pub fn clear_error(&self) {
        self.get().flags.fetch_and(!PAGE_ERROR, Ordering::SeqCst);
    }

    pub fn is_dirty(&self) -> bool {
        self.get().flags.load(Ordering::SeqCst) & PAGE_DIRTY != 0
    }

    pub fn set_dirty(&self) {
        tracing::debug!("set_dirty(page={})", self.get().id);
        self.get().flags.fetch_or(PAGE_DIRTY, Ordering::SeqCst);
    }

    pub fn clear_dirty(&self) {
        tracing::debug!("clear_dirty(page={})", self.get().id);
        self.get().flags.fetch_and(!PAGE_DIRTY, Ordering::SeqCst);
    }

    pub fn is_loaded(&self) -> bool {
        self.get().flags.load(Ordering::SeqCst) & PAGE_LOADED != 0
    }

    pub fn set_loaded(&self) {
        self.get().flags.fetch_or(PAGE_LOADED, Ordering::SeqCst);
    }

    pub fn clear_loaded(&self) {
        tracing::debug!("clear loaded {}", self.get().id);
        self.get().flags.fetch_and(!PAGE_LOADED, Ordering::SeqCst);
    }

    pub fn is_index(&self) -> bool {
        match self.get_contents().page_type() {
            PageType::IndexLeaf | PageType::IndexInterior => true,
            PageType::TableLeaf | PageType::TableInterior => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// The state of the current pager cache flush.
enum FlushState {
    /// Idle.
    Start,
    /// Waiting for all in-flight writes to the on-disk WAL to complete.
    WaitAppendFrames,
    /// Fsync the on-disk WAL.
    SyncWal,
    /// Checkpoint the WAL to the database file (if needed).
    Checkpoint,
    /// Fsync the database file.
    SyncDbFile,
    /// Waiting for the database file to be fsynced.
    WaitSyncDbFile,
}

#[derive(Clone, Debug, Copy)]
enum CheckpointState {
    Checkpoint,
    SyncDbFile,
    WaitSyncDbFile,
    CheckpointDone,
}

/// The mode of allocating a btree page.
pub enum BtreePageAllocMode {
    /// Allocate any btree page
    Any,
    /// Allocate a specific page number, typically used for root page allocation
    Exact(u32),
    /// Allocate a page number less than or equal to the parameter
    Le(u32),
}

/// This will keep track of the state of current cache flush in order to not repeat work
struct FlushInfo {
    state: FlushState,
    /// Number of writes taking place. When in_flight gets to 0 we can schedule a fsync.
    in_flight_writes: Rc<RefCell<usize>>,
}

pub const DB_STATE_UNITIALIZED: usize = 0;
pub const DB_STATE_INITIALIZING: usize = 1;
pub const DB_STATE_INITIALIZED: usize = 2;
/// The pager interface implements the persistence layer by providing access
/// to pages of the database file, including caching, concurrency control, and
/// transaction management.
pub struct Pager {
    /// Source of the database pages.
    pub db_file: Arc<dyn DatabaseStorage>,
    /// The write-ahead log (WAL) for the database.
    pub wal: Rc<RefCell<dyn Wal>>,
    /// A page cache for the database.
    pub page_cache: Arc<RwLock<DumbLruPageCache>>,
    /// Buffer pool for temporary data storage.
    pub buffer_pool: Rc<BufferPool>,
    /// I/O interface for input/output operations.
    pub io: Arc<dyn crate::io::IO>,
    pub dirty_pages: Rc<RefCell<HashSet<usize>>>,

    flush_info: RefCell<FlushInfo>,
    checkpoint_state: RefCell<CheckpointState>,
    checkpoint_inflight: Rc<RefCell<usize>>,
    syncing: Rc<RefCell<bool>>,

    reserved_space: OnceCell<u8>,
}

#[derive(Debug, Copy, Clone)]
/// The status of the current cache flush.
/// A Done state means that the WAL was committed to disk and fsynced,
/// plus potentially checkpointed to the DB (and the DB then fsynced).
pub enum PagerCacheflushStatus {
    Done(PagerCacheflushResult),
    IO,
}

#[derive(Debug, Copy, Clone)]
pub enum PagerCacheflushResult {
    /// The WAL was written to disk and fsynced.
    WalWritten,
    /// The WAL was written, fsynced, and a checkpoint was performed.
    /// The database file was then also fsynced.
    Checkpointed(CheckpointResult),
    Rollback,
}

#[derive(Clone)]
enum AllocatePage1State {
    Start,
    Writing {
        write_counter: Rc<RefCell<usize>>,
        page: BTreePage,
    },
    Done,
}

impl Pager {
    pub fn new(
        db_file: Arc<dyn DatabaseStorage>,
        wal: Rc<RefCell<dyn Wal>>,
        io: Arc<dyn crate::io::IO>,
        page_cache: Arc<RwLock<DumbLruPageCache>>,
        buffer_pool: Rc<BufferPool>,
    ) -> Result<Self> {
        Ok(Self {
            db_file,
            wal,
            page_cache,
            io,
            dirty_pages: Rc::new(RefCell::new(HashSet::new())),
            flush_info: RefCell::new(FlushInfo {
                state: FlushState::Start,
                in_flight_writes: Rc::new(RefCell::new(0)),
            }),
            syncing: Rc::new(RefCell::new(false)),
            checkpoint_state: RefCell::new(CheckpointState::Checkpoint),
            checkpoint_inflight: Rc::new(RefCell::new(0)),
            buffer_pool,
            reserved_space: OnceCell::new(),
        })
    }

    pub fn set_wal(&mut self, wal: Rc<RefCell<WalFile>>) {
        self.wal = wal;
    }

    pub fn maybe_allocate_page1(&self) -> Result<CursorResult<()>> {
        self.allocate_page1()?;
        Ok(CursorResult::Ok(()))
    }

    #[inline(always)]
    pub fn begin_write_tx(&self) -> Result<CursorResult<LimboResult>> {
        // TODO(Diego): The only possibly allocate page1 here is because OpenEphemeral needs a write transaction
        // we should have a unique API to begin transactions, something like sqlite3BtreeBeginTrans
        match self.maybe_allocate_page1()? {
            CursorResult::Ok(_) => {}
            CursorResult::IO => return Ok(CursorResult::IO),
        }
        Ok(CursorResult::Ok(self.wal.borrow_mut().begin_write_tx()?))
    }

    #[inline(always)]
    pub fn begin_read_tx(&self) -> Result<CursorResult<LimboResult>> {
        // We allocate the first page lazily in the first transaction
        match self.maybe_allocate_page1()? {
            CursorResult::Ok(_) => {}
            CursorResult::IO => return Ok(CursorResult::IO),
        }
        Ok(CursorResult::Ok(self.wal.borrow_mut().begin_read_tx()?))
    }

    pub fn end_tx(&self, rollback: bool) -> Result<PagerCacheflushStatus> {
        tracing::trace!("end_tx(rollback={})", rollback);
        if rollback {
            self.wal.borrow().end_write_tx()?;
            self.wal.borrow().end_read_tx()?;
            return Ok(PagerCacheflushStatus::Done(PagerCacheflushResult::Rollback));
        }
        let cacheflush_status = self.cacheflush()?;
        match cacheflush_status {
            PagerCacheflushStatus::IO => Ok(PagerCacheflushStatus::IO),
            PagerCacheflushStatus::Done(_) => {
                self.wal.borrow().end_write_tx()?;
                self.wal.borrow().end_read_tx()?;
                Ok(cacheflush_status)
            }
        }
    }

    pub fn end_read_tx(&self) -> Result<()> {
        self.wal.borrow().end_read_tx()?;
        Ok(())
    }

    /// Reads a page from the database.
    #[tracing::instrument(skip_all, level = Level::DEBUG)]
    pub fn read_page(&self, page_idx: usize) -> Result<PageRef, LimboError> {
        tracing::trace!("read_page(page_idx = {})", page_idx);
        let mut page_cache = self.page_cache.write();
        let page_key = PageCacheKey::new(page_idx);
        if let Some(page) = page_cache.get(&page_key) {
            tracing::trace!("read_page(page_idx = {}) = cached", page_idx);
            return Ok(page.clone());
        }
        let page = Arc::new(Page::new(page_idx));
        page.set_locked();

        if let Some(frame_id) = self.wal.borrow().find_frame(page_idx as u64)? {
            self.wal
                .borrow()
                .read_frame(frame_id, page.clone(), self.buffer_pool.clone())?;
            {
                page.set_uptodate();
            }
            // TODO(pere) should probably first insert to page cache, and if successful,
            // read frame or page
            match page_cache.insert(page_key, page.clone()) {
                Ok(_) => {}
                Err(CacheError::Full) => return Err(LimboError::CacheFull),
                Err(CacheError::KeyExists) => {
                    unreachable!("Page should not exist in cache after get() miss")
                }
                Err(e) => {
                    return Err(LimboError::InternalError(format!(
                        "Failed to insert page into cache: {:?}",
                        e
                    )))
                }
            }
            return Ok(page);
        }

        sqlite3_ondisk::begin_read_page(
            self.db_file.clone(),
            self.buffer_pool.clone(),
            page.clone(),
            page_idx,
        )?;
        match page_cache.insert(page_key, page.clone()) {
            Ok(_) => {}
            Err(CacheError::Full) => return Err(LimboError::CacheFull),
            Err(CacheError::KeyExists) => {
                unreachable!("Page should not exist in cache after get() miss")
            }
            Err(e) => {
                return Err(LimboError::InternalError(format!(
                    "Failed to insert page into cache: {:?}",
                    e
                )))
            }
        }
        Ok(page)
    }

    // Get a page from the cache, if it exists.
    pub fn cache_get(&self, page_idx: usize) -> Option<PageRef> {
        tracing::trace!("read_page(page_idx = {})", page_idx);
        let mut page_cache = self.page_cache.write();
        let page_key = PageCacheKey::new(page_idx);
        page_cache.get(&page_key)
    }

    /// Changes the size of the page cache.
    pub fn change_page_cache_size(&self, capacity: usize) -> Result<CacheResizeResult> {
        let mut page_cache = self.page_cache.write();
        Ok(page_cache.resize(capacity))
    }

    pub fn add_dirty(&self, page_id: usize) {
        // TODO: check duplicates?
        let mut dirty_pages = RefCell::borrow_mut(&self.dirty_pages);
        dirty_pages.insert(page_id);
    }

    pub fn wal_frame_count(&self) -> Result<u64> {
        Ok(self.wal.borrow().get_max_frame_in_wal())
    }

    /// Flush dirty pages to disk.
    /// In the base case, it will write the dirty pages to the WAL and then fsync the WAL.
    /// If the WAL size is over the checkpoint threshold, it will checkpoint the WAL to
    /// the database file and then fsync the database file.
    pub fn cacheflush(&self) -> Result<PagerCacheflushStatus> {
        let mut checkpoint_result = CheckpointResult::default();
        loop {
            let state = self.flush_info.borrow().state;
            trace!("cacheflush {:?}", state);
            match state {
                FlushState::Start => {
                    let db_size = header_accessor::get_database_size(self)?;
                    for (dirty_page_idx, page_id) in self.dirty_pages.borrow().iter().enumerate() {
                        let is_last_frame = dirty_page_idx == self.dirty_pages.borrow().len() - 1;
                        let mut cache = self.page_cache.write();
                        let page_key = PageCacheKey::new(*page_id);
                        let page = cache.get(&page_key).expect("we somehow added a page to dirty list but we didn't mark it as dirty, causing cache to drop it.");
                        let page_type = page.get().contents.as_ref().unwrap().maybe_page_type();
                        trace!("cacheflush(page={}, page_type={:?}", page_id, page_type);
                        let db_size = if is_last_frame { db_size } else { 0 };
                        self.wal.borrow_mut().append_frame(
                            page.clone(),
                            db_size,
                            self.flush_info.borrow().in_flight_writes.clone(),
                        )?;
                        page.clear_dirty();
                    }
                    // This is okay assuming we use shared cache by default.
                    {
                        let mut cache = self.page_cache.write();
                        cache.clear().unwrap();
                    }
                    self.dirty_pages.borrow_mut().clear();
                    self.flush_info.borrow_mut().state = FlushState::WaitAppendFrames;
                    return Ok(PagerCacheflushStatus::IO);
                }
                FlushState::WaitAppendFrames => {
                    let in_flight = *self.flush_info.borrow().in_flight_writes.borrow();
                    if in_flight == 0 {
                        self.flush_info.borrow_mut().state = FlushState::SyncWal;
                        self.wal.borrow_mut().finish_append_frames_commit()?;
                    } else {
                        return Ok(PagerCacheflushStatus::IO);
                    }
                }
                FlushState::SyncWal => {
                    if WalFsyncStatus::IO == self.wal.borrow_mut().sync()? {
                        return Ok(PagerCacheflushStatus::IO);
                    }

                    if !self.wal.borrow().should_checkpoint() {
                        self.flush_info.borrow_mut().state = FlushState::Start;
                        return Ok(PagerCacheflushStatus::Done(
                            PagerCacheflushResult::WalWritten,
                        ));
                    }
                    self.flush_info.borrow_mut().state = FlushState::Checkpoint;
                }
                FlushState::Checkpoint => {
                    match self.checkpoint()? {
                        CheckpointStatus::Done(res) => {
                            checkpoint_result = res;
                            self.flush_info.borrow_mut().state = FlushState::SyncDbFile;
                        }
                        CheckpointStatus::IO => return Ok(PagerCacheflushStatus::IO),
                    };
                }
                FlushState::SyncDbFile => {
                    sqlite3_ondisk::begin_sync(self.db_file.clone(), self.syncing.clone())?;
                    self.flush_info.borrow_mut().state = FlushState::WaitSyncDbFile;
                }
                FlushState::WaitSyncDbFile => {
                    if *self.syncing.borrow() {
                        return Ok(PagerCacheflushStatus::IO);
                    } else {
                        self.flush_info.borrow_mut().state = FlushState::Start;
                        break;
                    }
                }
            }
        }
        Ok(PagerCacheflushStatus::Done(
            PagerCacheflushResult::Checkpointed(checkpoint_result),
        ))
    }

    pub fn wal_get_frame(
        &self,
        frame_no: u32,
        p_frame: *mut u8,
        frame_len: u32,
    ) -> Result<Arc<Completion>> {
        let wal = self.wal.borrow();
        wal.read_frame_raw(
            frame_no.into(),
            self.buffer_pool.clone(),
            p_frame,
            frame_len,
        )
    }

    pub fn checkpoint(&self) -> Result<CheckpointStatus> {
        let mut checkpoint_result = CheckpointResult::default();
        loop {
            let state = *self.checkpoint_state.borrow();
            trace!("pager_checkpoint(state={:?})", state);
            match state {
                CheckpointState::Checkpoint => {
                    let in_flight = self.checkpoint_inflight.clone();
                    match self.wal.borrow_mut().checkpoint(
                        self,
                        in_flight,
                        CheckpointMode::Passive,
                    )? {
                        CheckpointStatus::IO => return Ok(CheckpointStatus::IO),
                        CheckpointStatus::Done(res) => {
                            checkpoint_result = res;
                            self.checkpoint_state.replace(CheckpointState::SyncDbFile);
                        }
                    };
                }
                CheckpointState::SyncDbFile => {
                    sqlite3_ondisk::begin_sync(self.db_file.clone(), self.syncing.clone())?;
                    self.checkpoint_state
                        .replace(CheckpointState::WaitSyncDbFile);
                }
                CheckpointState::WaitSyncDbFile => {
                    if *self.syncing.borrow() {
                        return Ok(CheckpointStatus::IO);
                    } else {
                        self.checkpoint_state
                            .replace(CheckpointState::CheckpointDone);
                    }
                }
                CheckpointState::CheckpointDone => {
                    return if *self.checkpoint_inflight.borrow() > 0 {
                        Ok(CheckpointStatus::IO)
                    } else {
                        self.checkpoint_state.replace(CheckpointState::Checkpoint);
                        Ok(CheckpointStatus::Done(checkpoint_result))
                    };
                }
            }
        }
    }

    /// Invalidates entire page cache by removing all dirty and clean pages. Usually used in case
    /// of a rollback or in case we want to invalidate page cache after starting a read transaction
    /// right after new writes happened which would invalidate current page cache.
    pub fn clear_page_cache(&self) {
        self.dirty_pages.borrow_mut().clear();
        self.page_cache.write().unset_dirty_all_pages();
        self.page_cache
            .write()
            .clear()
            .expect("Failed to clear page cache");
    }

    pub fn checkpoint_shutdown(&self) -> Result<()> {
        let mut attempts = 0;
        {
            let mut wal = self.wal.borrow_mut();
            // fsync the wal syncronously before beginning checkpoint
            while let Ok(WalFsyncStatus::IO) = wal.sync() {
                if attempts >= 10 {
                    return Err(LimboError::InternalError(
                        "Failed to fsync WAL before final checkpoint, fd likely closed".into(),
                    ));
                }
                self.io.run_once()?;
                attempts += 1;
            }
        }
        self.wal_checkpoint()?;
        Ok(())
    }

    pub fn wal_checkpoint(&self) -> Result<CheckpointResult> {
        let checkpoint_result: CheckpointResult;
        loop {
            match self.wal.borrow_mut().checkpoint(
                self,
                Rc::new(RefCell::new(0)),
                CheckpointMode::Passive,
            ) {
                Ok(CheckpointStatus::IO) => {
                    let _ = self.io.run_once();
                }
                Ok(CheckpointStatus::Done(res)) => {
                    checkpoint_result = res;
                    break;
                }
                Err(err) => panic!("error while clearing cache {}", err),
            }
        }
        // TODO: only clear cache of things that are really invalidated
        self.page_cache.write().clear().map_err(|e| {
            LimboError::InternalError(format!("Failed to clear page cache: {:?}", e))
        })?;
        Ok(checkpoint_result)
    }

    pub fn allocate_page1(&self) -> Result<CursorResult<PageRef>> {
        tracing::trace!("allocate_page1(Start)");
        let mut default_header = DatabaseHeader::default();
        default_header.database_size += 1;
        let page = self.allocate_page(1, 0);

        let contents = page.get_contents();
        contents.write_database_header(&default_header);

        let page1 = Arc::new(BTreePageInner::new(page));
        // Create the sqlite_schema table, for this we just need to create the btree page
        // for the first page of the database which is basically like any other btree page
        // but with a 100 byte offset, so we just init the page so that sqlite understands
        // this is a correct page.
        page1.init(
            PageType::TableLeaf,
            DATABASE_HEADER_SIZE,
            (default_header.get_page_size() - default_header.reserved_space as u32) as u16,
        );
        let write_counter = Rc::new(RefCell::new(0));
        begin_write_btree_page(self, &page1.get(), write_counter.clone())?;

        Ok(CursorResult::Ok(page1.get()))
    }

    pub fn update_dirty_loaded_page_in_cache(
        &self,
        id: usize,
        page: PageRef,
    ) -> Result<(), LimboError> {
        let mut cache = self.page_cache.write();
        let page_key = PageCacheKey::new(id);

        // FIXME: use specific page key for writer instead of max frame, this will make readers not conflict
        assert!(page.is_dirty());
        cache
            .insert_ignore_existing(page_key, page.clone())
            .map_err(|e| {
                LimboError::InternalError(format!(
                    "Failed to insert loaded page {} into cache: {:?}",
                    id, e
                ))
            })?;
        page.set_loaded();
        Ok(())
    }

    pub fn usable_size(&self) -> usize {
        let page_size = header_accessor::get_page_size(self).unwrap_or_default() as u32;
        let reserved_space = header_accessor::get_reserved_space(self).unwrap_or_default() as u32;
        (page_size - reserved_space) as usize
    }

    pub fn rollback(&self) -> Result<(), LimboError> {
        self.dirty_pages.borrow_mut().clear();
        let mut cache = self.page_cache.write();
        cache.unset_dirty_all_pages();
        cache.clear().expect("failed to clear page cache");
        self.wal.borrow_mut().rollback()?;

        Ok(())
    }

    pub fn allocate_page(&self, page_id: usize, offset: usize) -> PageRef {
        let page = Arc::new(Page::new(page_id));
        {
            let buffer = self.buffer_pool.get();
            let bp = self.buffer_pool.clone();
            let drop_fn = Rc::new(move |buf| {
                bp.put(buf);
            });
            let buffer = Arc::new(RefCell::new(Buffer::new(buffer, drop_fn)));
            page.set_loaded();
            page.get().contents = Some(PageContent::new(offset, buffer));
        }
        page
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::storage::page_cache::{DumbLruPageCache, PageCacheKey};

    use super::Page;

    #[test]
    fn test_shared_cache() {
        // ensure cache can be shared between threads
        let cache = Arc::new(RwLock::new(DumbLruPageCache::new(10)));

        let thread = {
            let cache = cache.clone();
            std::thread::spawn(move || {
                let mut cache = cache.write();
                let page_key = PageCacheKey::new(1);
                cache.insert(page_key, Arc::new(Page::new(1))).unwrap();
            })
        };
        let _ = thread.join();
        let mut cache = cache.write();
        let page_key = PageCacheKey::new(1);
        let page = cache.get(&page_key);
        assert_eq!(page.unwrap().get().id, 1);
    }
}
