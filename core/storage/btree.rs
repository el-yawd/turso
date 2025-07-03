use parking_lot::RwLock;

use std::cell::OnceCell;

use crate::{
    storage::{
        header_accessor,
        page_cache::{CacheError, PageCacheKey},
        pager::{BtreePageAllocMode, Pager},
        sqlite3_ondisk::PageType,
    },
    BufferPool, DatabaseStorage, WalFile, WalFileShared, IO,
};
use ptrmap::*;

use crate::{types::CursorResult, LimboError, Result};

use std::{
    cell::{RefCell, UnsafeCell},
    rc::Rc,
    sync::Arc,
};

#[cfg(not(feature = "omit_autovacuum"))]
use crate::io::Buffer as IoBuffer;

use super::btree_cursor::BTreeCursor;
use super::{page_cache::DumbLruPageCache, pager::PageRef, wal::DummyWAL};

/// The B-Tree page header is 12 bytes for interior pages and 8 bytes for leaf pages.
///
/// +--------+-----------------+-----------------+-----------------+--------+----- ..... ----+
/// | Page   | First Freeblock | Cell Count      | Cell Content    | Frag.  | Right-most     |
/// | Type   | Offset          |                 | Area Start      | Bytes  | pointer        |
/// +--------+-----------------+-----------------+-----------------+--------+----- ..... ----+
///     0        1        2        3        4        5        6        7        8       11
///
pub mod offset {
    /// Type of the B-Tree page (u8).
    pub const BTREE_PAGE_TYPE: usize = 0;

    /// A pointer to the first freeblock (u16).
    ///
    /// This field of the B-Tree page header is an offset to the first freeblock, or zero if
    /// there are no freeblocks on the page.  A freeblock is a structure used to identify
    /// unallocated space within a B-Tree page, organized as a chain.
    ///
    /// Please note that freeblocks do not mean the regular unallocated free space to the left
    /// of the cell content area pointer, but instead blocks of at least 4
    /// bytes WITHIN the cell content area that are not in use due to e.g.
    /// deletions.
    pub const BTREE_FIRST_FREEBLOCK: usize = 1;

    /// The number of cells in the page (u16).
    pub const BTREE_CELL_COUNT: usize = 3;

    /// A pointer to first byte of cell allocated content from top (u16).
    ///
    /// SQLite strives to place cells as far toward the end of the b-tree page as it can, in
    /// order to leave space for future growth of the cell pointer array. This means that the
    /// cell content area pointer moves leftward as cells are added to the page.
    pub const BTREE_CELL_CONTENT_AREA: usize = 5;

    /// The number of fragmented bytes (u8).
    ///
    /// Fragments are isolated groups of 1, 2, or 3 unused bytes within the cell content area.
    pub const BTREE_FRAGMENTED_BYTES_COUNT: usize = 7;

    /// The right-most pointer (saved separately from cells) (u32)
    pub const BTREE_RIGHTMOST_PTR: usize = 8;
}

/// Wrapper around a page reference used in order to update the reference in case page was unloaded
/// and we need to update the reference.
pub struct BTreePageInner {
    pub page: RefCell<PageRef>,
}

impl BTreePageInner {
    pub fn new(page: PageRef) -> Self {
        Self {
            page: RefCell::new(page),
        }
    }

    pub fn init(&self, page_type: PageType, offset: usize, usable_space: u16) {
        // setup btree page
        let contents = self.get();
        tracing::debug!("init_page(id={}, offset={})", contents.get().id, offset);
        let contents = contents.get().contents.as_mut().unwrap();
        contents.offset = offset;
        let id = page_type as u8;
        contents.write_u8(offset::BTREE_PAGE_TYPE, id);
        contents.write_u16(offset::BTREE_FIRST_FREEBLOCK, 0);
        contents.write_u16(offset::BTREE_CELL_COUNT, 0);

        contents.write_u16(offset::BTREE_CELL_CONTENT_AREA, usable_space);

        contents.write_u8(offset::BTREE_FRAGMENTED_BYTES_COUNT, 0);
        contents.write_u32(offset::BTREE_RIGHTMOST_PTR, 0);
    }

    pub fn get(&self) -> PageRef {
        self.page.borrow().clone()
    }
}
pub type BTreePage = Arc<BTreePageInner>;
unsafe impl Send for BTreePageInner {}
unsafe impl Sync for BTreePageInner {}

/// Track the state of the auto-vacuum mode.
#[derive(Clone, Copy, Debug)]
pub enum AutoVacuumMode {
    None,
    Full,
    Incremental,
}

#[derive(Debug)]
pub struct CreateBTreeFlags(pub u8);
impl CreateBTreeFlags {
    pub const TABLE: u8 = 0b0001;
    pub const INDEX: u8 = 0b0010;

    pub fn new_table() -> Self {
        Self(CreateBTreeFlags::TABLE)
    }

    pub fn new_index() -> Self {
        Self(CreateBTreeFlags::INDEX)
    }

    pub fn is_table(&self) -> bool {
        (self.0 & CreateBTreeFlags::TABLE) != 0
    }

    pub fn is_index(&self) -> bool {
        (self.0 & CreateBTreeFlags::INDEX) != 0
    }

    pub fn get_flags(&self) -> u8 {
        self.0
    }
}

/// A database connection contains a reference to an instance of this object for
/// **every database file that it has open**.
///
/// NOTE: Unlike SQLite, Turso doesn't have support for shared cache since it's a
/// obsolete feature (check: https://sqlite.org/sharedcache.html).
pub struct BTree {
    pub pager: Rc<Pager>,
    pub io: Arc<dyn IO>,
    /// List of all open cursors
    cursors: Vec<Arc<BTreeCursor>>,
    n_pages: usize,
    inTransaction: InTransaction,
    auto_vacuum_mode: RefCell<AutoVacuumMode>,
    /// Cache page_size and reserved_space at Pager init and reuse for subsequent
    /// `usable_space` calls. TODO: Invalidate reserved_space when we add the functionality
    /// to change it.
    page_size: OnceCell<u16>,
    reserved_space: OnceCell<u8>,
    // maybe add the double-linked list of dbs?
}

impl BTree {
    // Open a database file
    pub fn open(
        io: Arc<dyn IO>,
        wal: Option<Arc<UnsafeCell<WalFileShared>>>,
        db_file: Arc<dyn DatabaseStorage>,
    ) -> Result<Rc<Self>> {
        let buffer_pool = Rc::new(BufferPool::new(None));

        // Open existing WAL file if present
        if let Some(shared_wal) = wal {
            // No pages in DB file or WAL -> empty database
            let wal = Rc::new(RefCell::new(WalFile::new(
                io.clone(),
                shared_wal,
                buffer_pool.clone(),
            )));
            let pager = Rc::new(Pager::new(
                db_file.clone(),
                wal,
                io.clone(),
                Arc::new(RwLock::new(DumbLruPageCache::default())),
                buffer_pool,
            )?);

            let page_size = header_accessor::get_page_size(&pager)
                .unwrap_or(crate::storage::sqlite3_ondisk::DEFAULT_PAGE_SIZE)
                as u32;

            pager.buffer_pool.set_page_size(page_size as usize);

            return Ok(Rc::new(Self {
                pager,
                io,
                cursors: Vec::new(),
                n_pages: 0,
                inTransaction: InTransaction::None,
                auto_vacuum_mode: RefCell::new(AutoVacuumMode::None),
                page_size: OnceCell::new(),
                reserved_space: OnceCell::new(),
            }));
        };

        // No existing WAL; create one.
        // TODO: currently Pager needs to be instantiated with some implementation of trait Wal, so here's a workaround.
        let dummy_wal = Rc::new(RefCell::new(DummyWAL {}));
        let pager = Rc::new(Pager::new(
            db_file.clone(),
            dummy_wal,
            io.clone(),
            Arc::new(RwLock::new(DumbLruPageCache::default())),
            buffer_pool.clone(),
        )?);

        Ok(Rc::new(Self {
            pager,
            io,
            cursors: Vec::new(),
            n_pages: 0,
            inTransaction: InTransaction::None,

            auto_vacuum_mode: RefCell::new(AutoVacuumMode::None),
            page_size: OnceCell::new(),

            reserved_space: OnceCell::new(),
        }))
    }

    pub fn get_auto_vacuum_mode(&self) -> AutoVacuumMode {
        *self.auto_vacuum_mode.borrow()
    }

    pub fn set_auto_vacuum_mode(&self, mode: AutoVacuumMode) {
        *self.auto_vacuum_mode.borrow_mut() = mode;
    }

    /// This method is used to allocate a new root page for a btree, both for tables and indexes
    // FIXME: handle no room in page cache
    pub fn create(&self, flags: &CreateBTreeFlags) -> Result<CursorResult<u32>> {
        let page_type = match flags {
            _ if flags.is_table() => PageType::TableLeaf,
            _ if flags.is_index() => PageType::IndexLeaf,
            _ => unreachable!("Invalid flags state"),
        };
        #[cfg(feature = "omit_autovacuum")]
        {
            let page = self.do_allocate_page(page_type, 0, BtreePageAllocMode::Any);
            let page_id = page.get().get().id;
            Ok(CursorResult::Ok(page_id as u32))
        }

        //  If autovacuum is enabled, we need to allocate a new page number that is greater than the largest root page number
        #[cfg(not(feature = "omit_autovacuum"))]
        {
            let auto_vacuum_mode = self.auto_vacuum_mode.borrow();
            match *auto_vacuum_mode {
                AutoVacuumMode::None => {
                    let page = self.do_allocate_page(page_type, 0, BtreePageAllocMode::Any);
                    let page_id = page.get().get().id;
                    Ok(CursorResult::Ok(page_id as u32))
                }
                AutoVacuumMode::Full => {
                    let mut root_page_num =
                        header_accessor::get_vacuum_mode_largest_root_page(&self.pager)?;
                    assert!(root_page_num > 0); //  Largest root page number cannot be 0 because that is set to 1 when creating the database with autovacuum enabled
                    root_page_num += 1;
                    assert!(root_page_num >= FIRST_PTRMAP_PAGE_NO); //  can never be less than 2 because we have already incremented

                    while is_ptrmap_page(
                        root_page_num,
                        header_accessor::get_page_size(&self.pager)? as usize,
                    ) {
                        root_page_num += 1;
                    }
                    assert!(root_page_num >= 3); //  the very first root page is page 3

                    //  root_page_num here is the desired root page
                    let page = self.do_allocate_page(
                        page_type,
                        0,
                        BtreePageAllocMode::Exact(root_page_num),
                    );
                    let allocated_page_id = page.get().get().id as u32;
                    if allocated_page_id != root_page_num {
                        //  TODO(Zaid): Handle swapping the allocated page with the desired root page
                    }

                    //  TODO(Zaid): Update the header metadata to reflect the new root page number

                    //  For now map allocated_page_id since we are not swapping it with root_page_num
                    match self.ptrmap_put(allocated_page_id, PtrmapType::RootPage, 0)? {
                        CursorResult::Ok(_) => Ok(CursorResult::Ok(allocated_page_id as u32)),
                        CursorResult::IO => Ok(CursorResult::IO),
                    }
                }
                AutoVacuumMode::Incremental => {
                    unimplemented!()
                }
            }
        }
    }

    /**
        Gets a new page that increasing the size of the page or uses a free page.
        Currently free list pages are not yet supported.
    */
    // FIXME: handle no room in page cache
    #[allow(clippy::readonly_write_lock)]
    pub fn allocate_page(&self) -> Result<PageRef> {
        let old_db_size = header_accessor::get_database_size(&self.pager)?;
        #[allow(unused_mut)]
        let mut new_db_size = old_db_size + 1;

        tracing::debug!("allocate_page(database_size={})", new_db_size);

        #[cfg(not(feature = "omit_autovacuum"))]
        {
            //  If the following conditions are met, allocate a pointer map page, add to cache and increment the database size
            //  - autovacuum is enabled
            //  - the last page is a pointer map page
            if matches!(*self.auto_vacuum_mode.borrow(), AutoVacuumMode::Full)
                && is_ptrmap_page(
                    new_db_size,
                    header_accessor::get_page_size(&self.pager)? as usize,
                )
            {
                let page = self.pager.allocate_page(new_db_size as usize, 0);
                page.set_dirty();
                self.pager.add_dirty(page.get().id);

                let page_key = PageCacheKey::new(page.get().id);
                let mut cache = self.pager.page_cache.write();
                match cache.insert(page_key, page.clone()) {
                    Ok(_) => (),
                    Err(CacheError::Full) => return Err(LimboError::CacheFull),
                    Err(_) => {
                        return Err(LimboError::InternalError(
                            "Unknown error inserting page to cache".into(),
                        ))
                    }
                }
                // we allocated a ptrmap page, so the next data page will be at new_db_size + 1
                new_db_size += 1;
            }
        }

        header_accessor::set_database_size(&self.pager, new_db_size)?;

        // FIXME: should reserve page cache entry before modifying the database
        let page = self.pager.allocate_page(new_db_size as usize, 0);
        {
            // setup page and add to cache
            page.set_dirty();
            self.pager.add_dirty(page.get().id);

            let page_key = PageCacheKey::new(page.get().id);
            let mut cache = self.pager.page_cache.write();
            match cache.insert(page_key, page.clone()) {
                Err(CacheError::Full) => Err(LimboError::CacheFull),
                Err(_) => Err(LimboError::InternalError(
                    "Unknown error inserting page to cache".into(),
                )),
                Ok(_) => Ok(page),
            }
        }
    }

    /// Allocate a new overflow page.
    /// This is done when a cell overflows and new space is needed.
    // FIXME: handle no room in page cache
    pub fn allocate_overflow_page(&self) -> PageRef {
        let page = self.allocate_page().unwrap();
        tracing::debug!("Pager::allocate_overflow_page(id={})", page.get().id);

        // setup overflow page
        let contents = page.get().contents.as_mut().unwrap();
        let buf = contents.as_ptr();
        buf.fill(0);

        page
    }

    /// Allocate a new page to the btree via the pager.
    /// This marks the page as dirty and writes the page header.
    // FIXME: handle no room in page cache
    pub fn do_allocate_page(
        &self,
        page_type: PageType,
        offset: usize,
        _alloc_mode: BtreePageAllocMode,
    ) -> BTreePage {
        let page = self.allocate_page().unwrap();
        let page = Arc::new(BTreePageInner::new(page));
        page.init(page_type, offset, self.usable_space() as u16);
        tracing::debug!(
            "do_allocate_page(id={}, page_type={:?})",
            page.get().get().id,
            page.get().get_contents().page_type()
        );
        page
    }

    pub fn usable_size(&self) -> usize {
        let page_size = header_accessor::get_page_size(&self.pager).unwrap_or_default() as u32;
        let reserved_space =
            header_accessor::get_reserved_space(&self.pager).unwrap_or_default() as u32;
        (page_size - reserved_space) as usize
    }

    /// The "usable size" of a database page is the page size specified by the 2-byte integer at offset 16
    /// in the header, minus the "reserved" space size recorded in the 1-byte integer at offset 20 in the header.
    /// The usable size of a page might be an odd number. However, the usable size is not allowed to be less than 480.
    /// In other words, if the page size is 512, then the reserved space size cannot exceed 32.
    pub fn usable_space(&self) -> usize {
        let page_size = *self
            .page_size
            .get_or_init(|| header_accessor::get_page_size(&self.pager).unwrap_or_default());

        let reserved_space = *self
            .reserved_space
            .get_or_init(|| header_accessor::get_reserved_space(&self.pager).unwrap_or_default());

        (page_size as usize) - (reserved_space as usize)
    }

    // Providing a page is optional, if provided it will be used to avoid reading the page from disk.
    // This is implemented in accordance with sqlite freepage2() function.
    pub fn free_page(&self, page: Option<PageRef>, page_id: usize) -> Result<()> {
        tracing::trace!("free_page(page_id={})", page_id);
        const TRUNK_PAGE_HEADER_SIZE: usize = 8;
        const LEAF_ENTRY_SIZE: usize = 4;
        const RESERVED_SLOTS: usize = 2;

        const TRUNK_PAGE_NEXT_PAGE_OFFSET: usize = 0; // Offset to next trunk page pointer
        const TRUNK_PAGE_LEAF_COUNT_OFFSET: usize = 4; // Offset to leaf count

        if page_id < 2 || page_id > header_accessor::get_database_size(&self.pager)? as usize {
            return Err(LimboError::Corrupt(format!(
                "Invalid page number {} for free operation",
                page_id
            )));
        }

        let page = match page {
            Some(page) => {
                assert_eq!(page.get().id, page_id, "Page id mismatch");
                page
            }
            None => self.pager.read_page(page_id)?,
        };

        header_accessor::set_freelist_pages(
            &self.pager,
            header_accessor::get_freelist_pages(&self.pager)? + 1,
        )?;

        let trunk_page_id = header_accessor::get_freelist_trunk_page(&self.pager)?;

        if trunk_page_id != 0 {
            // Add as leaf to current trunk
            let trunk_page = self.pager.read_page(trunk_page_id as usize)?;
            let trunk_page_contents = trunk_page.get().contents.as_ref().unwrap();
            let number_of_leaf_pages = trunk_page_contents.read_u32(TRUNK_PAGE_LEAF_COUNT_OFFSET);

            // Reserve 2 slots for the trunk page header which is 8 bytes or 2*LEAF_ENTRY_SIZE
            let max_free_list_entries = (self.usable_space() / LEAF_ENTRY_SIZE) - RESERVED_SLOTS;

            if number_of_leaf_pages < max_free_list_entries as u32 {
                trunk_page.set_dirty();
                self.pager.add_dirty(trunk_page_id as usize);

                trunk_page_contents
                    .write_u32(TRUNK_PAGE_LEAF_COUNT_OFFSET, number_of_leaf_pages + 1);
                trunk_page_contents.write_u32(
                    TRUNK_PAGE_HEADER_SIZE + (number_of_leaf_pages as usize * LEAF_ENTRY_SIZE),
                    page_id as u32,
                );
                page.clear_uptodate();
                page.clear_loaded();

                return Ok(());
            }
        }

        // If we get here, need to make this page a new trunk
        page.set_dirty();
        self.pager.add_dirty(page_id);

        let contents = page.get().contents.as_mut().unwrap();
        // Point to previous trunk
        contents.write_u32(TRUNK_PAGE_NEXT_PAGE_OFFSET, trunk_page_id);
        // Zero leaf count
        contents.write_u32(TRUNK_PAGE_LEAF_COUNT_OFFSET, 0);
        // Update page 1 to point to new trunk
        header_accessor::set_freelist_trunk_page(&self.pager, page_id as u32)?;
        // Clear flags
        page.clear_uptodate();
        page.clear_loaded();
        Ok(())
    }

    // TODO: Convert PageRef to BTreePage
    pub fn read_page(&self, page_idx: usize) -> Result<PageRef> {
        self.pager.read_page(page_idx)
    }

    // awful name, BTreePages should be handled by BTree and PageRef by Pager
    pub fn do_read_page(&self, page_idx: usize) -> Result<BTreePage> {
        self.pager
            .read_page(page_idx)
            .map(|page| Arc::new(BTreePageInner::new(page)))
    }

    /// Retrieves the pointer map entry for a given database page.
    /// `target_page_num` (1-indexed) is the page whose entry is sought.
    /// Returns `Ok(None)` if the page is not supposed to have a ptrmap entry (e.g. header, or a ptrmap page itself).
    #[cfg(not(feature = "omit_autovacuum"))]
    pub fn ptrmap_get(&self, target_page_num: u32) -> Result<CursorResult<Option<PtrmapEntry>>> {
        use crate::storage::{
            btree::ptrmap::{
                get_ptrmap_offset_in_page, get_ptrmap_page_no_for_db_page, is_ptrmap_page,
                FIRST_PTRMAP_PAGE_NO,
            },
            sqlite3_ondisk::PageContent,
        };

        tracing::trace!("ptrmap_get(page_idx = {})", target_page_num);
        let configured_page_size = header_accessor::get_page_size(&self.pager)? as usize;

        if target_page_num < FIRST_PTRMAP_PAGE_NO
            || is_ptrmap_page(target_page_num, configured_page_size)
        {
            return Ok(CursorResult::Ok(None));
        }

        let ptrmap_pg_no = get_ptrmap_page_no_for_db_page(target_page_num, configured_page_size);
        let offset_in_ptrmap_page =
            get_ptrmap_offset_in_page(target_page_num, ptrmap_pg_no, configured_page_size)?;
        tracing::trace!(
            "ptrmap_get(page_idx = {}) = ptrmap_pg_no = {}",
            target_page_num,
            ptrmap_pg_no
        );

        let ptrmap_page = self.read_page(ptrmap_pg_no as usize)?;
        if ptrmap_page.is_locked() {
            return Ok(CursorResult::IO);
        }
        if !ptrmap_page.is_loaded() {
            return Ok(CursorResult::IO);
        }
        let ptrmap_page_inner = ptrmap_page.get();

        let page_content: &PageContent = match ptrmap_page_inner.contents.as_ref() {
            Some(content) => content,
            None => {
                return Err(LimboError::InternalError(format!(
                    "Ptrmap page {} content not loaded",
                    ptrmap_pg_no
                )))
            }
        };

        let page_buffer_guard: std::cell::Ref<IoBuffer> = page_content.buffer.borrow();
        let full_buffer_slice: &[u8] = page_buffer_guard.as_slice();

        // Ptrmap pages are not page 1, so their internal offset within their buffer should be 0.
        // The actual page data starts at page_content.offset within the full_buffer_slice.
        if ptrmap_pg_no != 1 && page_content.offset != 0 {
            return Err(LimboError::Corrupt(format!(
                "Ptrmap page {} has unexpected internal offset {}",
                ptrmap_pg_no, page_content.offset
            )));
        }
        let ptrmap_page_data_slice: &[u8] = &full_buffer_slice[page_content.offset..];
        let actual_data_length = ptrmap_page_data_slice.len();

        // Check if the calculated offset for the entry is within the bounds of the actual page data length.
        if offset_in_ptrmap_page + PTRMAP_ENTRY_SIZE > actual_data_length {
            return Err(LimboError::InternalError(format!(
                "Ptrmap offset {} + entry size {} out of bounds for page {} (actual data len {})",
                offset_in_ptrmap_page, PTRMAP_ENTRY_SIZE, ptrmap_pg_no, actual_data_length
            )));
        }

        let entry_slice = &ptrmap_page_data_slice
            [offset_in_ptrmap_page..offset_in_ptrmap_page + PTRMAP_ENTRY_SIZE];
        match PtrmapEntry::deserialize(entry_slice) {
            Some(entry) => Ok(CursorResult::Ok(Some(entry))),
            None => Err(LimboError::Corrupt(format!(
                "Failed to deserialize ptrmap entry for page {} from ptrmap page {}",
                target_page_num, ptrmap_pg_no
            ))),
        }
    }

    /// Writes or updates the pointer map entry for a given database page.
    /// `db_page_no_to_update` (1-indexed) is the page whose entry is to be set.
    /// `entry_type` and `parent_page_no` define the new entry.
    #[cfg(not(feature = "omit_autovacuum"))]
    pub fn ptrmap_put(
        &self,
        db_page_no_to_update: u32,
        entry_type: PtrmapType,
        parent_page_no: u32,
    ) -> Result<CursorResult<()>> {
        tracing::trace!(
            "ptrmap_put(page_idx = {}, entry_type = {:?}, parent_page_no = {})",
            db_page_no_to_update,
            entry_type,
            parent_page_no
        );

        let page_size = header_accessor::get_page_size(&self.pager)? as usize;

        if db_page_no_to_update < FIRST_PTRMAP_PAGE_NO
            || is_ptrmap_page(db_page_no_to_update, page_size)
        {
            return Err(LimboError::InternalError(format!(
                "Cannot set ptrmap entry for page {}: it's a header/ptrmap page or invalid.",
                db_page_no_to_update
            )));
        }

        let ptrmap_pg_no = get_ptrmap_page_no_for_db_page(db_page_no_to_update, page_size);
        let offset_in_ptrmap_page =
            get_ptrmap_offset_in_page(db_page_no_to_update, ptrmap_pg_no, page_size)?;
        tracing::trace!(
                "ptrmap_put(page_idx = {}, entry_type = {:?}, parent_page_no = {}) = ptrmap_pg_no = {}, offset_in_ptrmap_page = {}",
                db_page_no_to_update,
                entry_type,
                parent_page_no,
                ptrmap_pg_no,
                offset_in_ptrmap_page
            );

        let ptrmap_page = self.read_page(ptrmap_pg_no as usize)?;
        if ptrmap_page.is_locked() {
            return Ok(CursorResult::IO);
        }
        if !ptrmap_page.is_loaded() {
            return Ok(CursorResult::IO);
        }
        let ptrmap_page_inner = ptrmap_page.get();

        let page_content = match ptrmap_page_inner.contents.as_ref() {
            Some(content) => content,
            None => {
                return Err(LimboError::InternalError(format!(
                    "Ptrmap page {} content not loaded",
                    ptrmap_pg_no
                )))
            }
        };

        let mut page_buffer_guard = page_content.buffer.borrow_mut();
        let full_buffer_slice = page_buffer_guard.as_mut_slice();

        if offset_in_ptrmap_page + PTRMAP_ENTRY_SIZE > full_buffer_slice.len() {
            return Err(LimboError::InternalError(format!(
                "Ptrmap offset {} + entry size {} out of bounds for page {} (actual data len {})",
                offset_in_ptrmap_page,
                PTRMAP_ENTRY_SIZE,
                ptrmap_pg_no,
                full_buffer_slice.len()
            )));
        }

        let entry = PtrmapEntry {
            entry_type,
            parent_page_no,
        };
        entry.serialize(
            &mut full_buffer_slice
                [offset_in_ptrmap_page..offset_in_ptrmap_page + PTRMAP_ENTRY_SIZE],
        )?;

        ptrmap_page.set_dirty();
        self.pager.add_dirty(ptrmap_pg_no as usize);
        Ok(CursorResult::Ok(()))
    }

    pub fn update_auto_vacuum_mode(
        &self,
        auto_vacuum_mode: AutoVacuumMode,
        largest_root_page_number: u32,
    ) -> crate::Result<()> {
        header_accessor::set_vacuum_mode_largest_root_page(&self.pager, largest_root_page_number)?;
        self.set_auto_vacuum_mode(auto_vacuum_mode);
        Ok(())
    }
}

enum InTransaction {
    None,
    Read,
    Write,
}

/*
** The pointer map is a lookup table that identifies the parent page for
** each child page in the database file.  The parent page is the page that
** contains a pointer to the child.  Every page in the database contains
** 0 or 1 parent pages. Each pointer map entry consists of a single byte 'type'
** and a 4 byte parent page number.
**
** The PTRMAP_XXX identifiers below are the valid types.
**
** The purpose of the pointer map is to facilitate moving pages from one
** position in the file to another as part of autovacuum.  When a page
** is moved, the pointer in its parent must be updated to point to the
** new location.  The pointer map is used to locate the parent page quickly.
**
** PTRMAP_ROOTPAGE: The database page is a root-page. The page-number is not
**                  used in this case.
**
** PTRMAP_FREEPAGE: The database page is an unused (free) page. The page-number
**                  is not used in this case.
**
** PTRMAP_OVERFLOW1: The database page is the first page in a list of
**                   overflow pages. The page number identifies the page that
**                   contains the cell with a pointer to this overflow page.
**
** PTRMAP_OVERFLOW2: The database page is the second or later page in a list of
**                   overflow pages. The page-number identifies the previous
**                   page in the overflow page list.
**
** PTRMAP_BTREE: The database page is a non-root btree page. The page number
**               identifies the parent page in the btree.
*/
#[cfg(not(feature = "omit_autovacuum"))]
pub mod ptrmap {
    use crate::{storage::sqlite3_ondisk::MIN_PAGE_SIZE, LimboError, Result};

    // Constants
    pub const PTRMAP_ENTRY_SIZE: usize = 5;
    /// Page 1 is the schema page which contains the database header.
    /// Page 2 is the first pointer map page if the database has any pointer map pages.
    pub const FIRST_PTRMAP_PAGE_NO: u32 = 2;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum PtrmapType {
        RootPage = 1,
        FreePage = 2,
        Overflow1 = 3,
        Overflow2 = 4,
        BTreeNode = 5,
    }

    impl PtrmapType {
        pub fn from_u8(value: u8) -> Option<Self> {
            match value {
                1 => Some(PtrmapType::RootPage),
                2 => Some(PtrmapType::FreePage),
                3 => Some(PtrmapType::Overflow1),
                4 => Some(PtrmapType::Overflow2),
                5 => Some(PtrmapType::BTreeNode),
                _ => None,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct PtrmapEntry {
        pub entry_type: PtrmapType,
        pub parent_page_no: u32,
    }

    impl PtrmapEntry {
        pub fn serialize(&self, buffer: &mut [u8]) -> Result<()> {
            if buffer.len() < PTRMAP_ENTRY_SIZE {
                return Err(LimboError::InternalError(format!(
                "Buffer too small to serialize ptrmap entry. Expected at least {} bytes, got {}",
                PTRMAP_ENTRY_SIZE,
                buffer.len()
            )));
            }
            buffer[0] = self.entry_type as u8;
            buffer[1..5].copy_from_slice(&self.parent_page_no.to_be_bytes());
            Ok(())
        }

        pub fn deserialize(buffer: &[u8]) -> Option<Self> {
            if buffer.len() < PTRMAP_ENTRY_SIZE {
                return None;
            }
            let entry_type_u8 = buffer[0];
            let parent_bytes_slice = buffer.get(1..5)?;
            let parent_page_no = u32::from_be_bytes(parent_bytes_slice.try_into().ok()?);
            PtrmapType::from_u8(entry_type_u8).map(|entry_type| PtrmapEntry {
                entry_type,
                parent_page_no,
            })
        }
    }

    /// Calculates how many database pages are mapped by a single pointer map page.
    /// This is based on the total page size, as ptrmap pages are filled with entries.
    pub fn entries_per_ptrmap_page(page_size: usize) -> usize {
        assert!(page_size >= MIN_PAGE_SIZE as usize);
        page_size / PTRMAP_ENTRY_SIZE
    }

    /// Calculates the cycle length of pointer map pages
    /// The cycle length is the number of database pages that are mapped by a single pointer map page.
    pub fn ptrmap_page_cycle_length(page_size: usize) -> usize {
        assert!(page_size >= MIN_PAGE_SIZE as usize);
        (page_size / PTRMAP_ENTRY_SIZE) + 1
    }

    /// Determines if a given page number `db_page_no` (1-indexed) is a pointer map page in a database with autovacuum enabled
    pub fn is_ptrmap_page(db_page_no: u32, page_size: usize) -> bool {
        //  The first page cannot be a ptrmap page because its for the schema
        if db_page_no == 1 {
            return false;
        }
        if db_page_no == FIRST_PTRMAP_PAGE_NO {
            return true;
        }
        get_ptrmap_page_no_for_db_page(db_page_no, page_size) == db_page_no
    }

    /// Calculates which pointer map page (1-indexed) contains the entry for `db_page_no_to_query` (1-indexed).
    /// `db_page_no_to_query` is the page whose ptrmap entry we are interested in.
    pub fn get_ptrmap_page_no_for_db_page(db_page_no_to_query: u32, page_size: usize) -> u32 {
        let group_size = ptrmap_page_cycle_length(page_size) as u32;
        if group_size == 0 {
            panic!("Page size too small, a ptrmap page cannot map any db pages.");
        }

        let effective_page_index = db_page_no_to_query - FIRST_PTRMAP_PAGE_NO;
        let group_idx = effective_page_index / group_size;

        (group_idx * group_size) + FIRST_PTRMAP_PAGE_NO
    }

    /// Calculates the byte offset of the entry for `db_page_no_to_query` (1-indexed)
    /// within its pointer map page (`ptrmap_page_no`, 1-indexed).
    pub fn get_ptrmap_offset_in_page(
        db_page_no_to_query: u32,
        ptrmap_page_no: u32,
        page_size: usize,
    ) -> Result<usize> {
        // The data pages mapped by `ptrmap_page_no` are:
        // `ptrmap_page_no + 1`, `ptrmap_page_no + 2`, ..., up to `ptrmap_page_no + n_data_pages_per_group`.
        // `db_page_no_to_query` must be one of these.
        // The 0-indexed position of `db_page_no_to_query` within this sequence of data pages is:
        // `db_page_no_to_query - (ptrmap_page_no + 1)`.

        let n_data_pages_per_group = entries_per_ptrmap_page(page_size);
        let first_data_page_mapped = ptrmap_page_no + 1;
        let last_data_page_mapped = ptrmap_page_no + n_data_pages_per_group as u32;

        if db_page_no_to_query < first_data_page_mapped
            || db_page_no_to_query > last_data_page_mapped
        {
            return Err(LimboError::InternalError(format!(
                "Page {} is not mapped by the data page range [{}, {}] of ptrmap page {}",
                db_page_no_to_query, first_data_page_mapped, last_data_page_mapped, ptrmap_page_no
            )));
        }
        if is_ptrmap_page(db_page_no_to_query, page_size) {
            return Err(LimboError::InternalError(format!(
                "Page {} is a pointer map page and should not have an entry calculated this way.",
                db_page_no_to_query
            )));
        }

        let entry_index_on_page = (db_page_no_to_query - first_data_page_mapped) as usize;
        Ok(entry_index_on_page * PTRMAP_ENTRY_SIZE)
    }
}

#[cfg(test)]
#[cfg(not(feature = "omit_autovacuum"))]
mod ptrmap_tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use super::ptrmap::*;
    use super::*;
    use crate::io::{MemoryIO, OpenFlags, IO};
    use crate::storage::buffer_pool::BufferPool;
    use crate::storage::database::{DatabaseFile, DatabaseStorage};
    use crate::storage::page_cache::DumbLruPageCache;
    use crate::storage::pager::Pager;
    use crate::storage::sqlite3_ondisk::MIN_PAGE_SIZE;
    use crate::storage::wal::{WalFile, WalFileShared};

    pub fn run_until_done<T>(
        mut action: impl FnMut() -> Result<CursorResult<T>>,
        pager: &Pager,
    ) -> Result<T> {
        loop {
            match action()? {
                CursorResult::Ok(res) => {
                    return Ok(res);
                }
                CursorResult::IO => pager.io.run_once().unwrap(),
            }
        }
    }
    // Helper to create a BTree for testing
    fn test_btree_setup(page_size: u32, initial_db_pages: u32) -> Rc<BTree> {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let db_file: Arc<dyn DatabaseStorage> = Arc::new(DatabaseFile::new(
            io.open_file("test.db", OpenFlags::Create, true).unwrap(),
        ));

        //  Construct interfaces for the pager
        let buffer_pool = Rc::new(BufferPool::new(Some(page_size as usize)));
        let page_cache = Arc::new(RwLock::new(DumbLruPageCache::new(
            (initial_db_pages + 10) as usize,
        )));
        let shared_wal = WalFileShared::new_shared(
            page_size,
            &io.clone(),
            io.open_file("test.db-wal", OpenFlags::Create, false)
                .unwrap(),
        )
        .unwrap();

        let wal = Rc::new(RefCell::new(WalFile::new(
            io.clone(),
            shared_wal.clone(),
            buffer_pool.clone(),
        )));

        let btree = BTree::open(io.clone(), Some(shared_wal), db_file.clone()).unwrap();
        run_until_done(|| btree.pager.allocate_page1(), &btree.pager).unwrap();
        header_accessor::set_vacuum_mode_largest_root_page(&btree.pager, 1).unwrap();

        //  Allocate all the pages as btree root pages
        for _ in 0..initial_db_pages {
            match btree.create(&CreateBTreeFlags::new_table()) {
                Ok(CursorResult::Ok(_root_page_id)) => (),
                Ok(CursorResult::IO) => {
                    panic!("test_pager_setup: btree_create returned CursorResult::IO unexpectedly");
                }
                Err(e) => {
                    panic!("test_pager_setup: btree_create failed: {:?}", e);
                }
            }
        }

        btree
    }

    #[test]
    fn test_ptrmap_page_allocation() {
        let page_size = 4096;
        let initial_db_pages = 10;
        let btree = test_btree_setup(page_size, initial_db_pages);

        // Page 5 should be mapped by ptrmap page 2.
        let db_page_to_update: u32 = 5;
        let expected_ptrmap_pg_no =
            get_ptrmap_page_no_for_db_page(db_page_to_update, page_size as usize);
        assert_eq!(expected_ptrmap_pg_no, FIRST_PTRMAP_PAGE_NO);

        //  Ensure the pointer map page ref is created and loadable via the pager
        let ptrmap_page_ref = btree.pager.read_page(expected_ptrmap_pg_no as usize);
        assert!(ptrmap_page_ref.is_ok());

        //  Ensure that the database header size is correctly reflected
        assert_eq!(
            header_accessor::get_database_size(&btree.pager).unwrap(),
            initial_db_pages + 2
        ); // (1+1) -> (header + ptrmap)

        //  Read the entry from the ptrmap page and verify it
        let entry = btree.ptrmap_get(db_page_to_update).unwrap();
        assert!(matches!(entry, CursorResult::Ok(Some(_))));
        let CursorResult::Ok(Some(entry)) = entry else {
            panic!("entry is not Some");
        };
        assert_eq!(entry.entry_type, PtrmapType::RootPage);
        assert_eq!(entry.parent_page_no, 0);
    }

    #[test]
    fn test_is_ptrmap_page_logic() {
        let page_size = MIN_PAGE_SIZE as usize;
        let n_data_pages = entries_per_ptrmap_page(page_size);
        assert_eq!(n_data_pages, 102); //   512/5 = 102

        assert!(!is_ptrmap_page(1, page_size)); // Header
        assert!(is_ptrmap_page(2, page_size)); // P0
        assert!(!is_ptrmap_page(3, page_size)); // D0_1
        assert!(!is_ptrmap_page(4, page_size)); // D0_2
        assert!(!is_ptrmap_page(5, page_size)); // D0_3
        assert!(is_ptrmap_page(105, page_size)); // P1
        assert!(!is_ptrmap_page(106, page_size)); // D1_1
        assert!(!is_ptrmap_page(107, page_size)); // D1_2
        assert!(!is_ptrmap_page(108, page_size)); // D1_3
        assert!(is_ptrmap_page(208, page_size)); // P2
    }

    #[test]
    fn test_get_ptrmap_page_no() {
        let page_size = MIN_PAGE_SIZE as usize; // Maps 103 data pages

        // Test pages mapped by P0 (page 2)
        assert_eq!(get_ptrmap_page_no_for_db_page(3, page_size), 2); // D(3) -> P0(2)
        assert_eq!(get_ptrmap_page_no_for_db_page(4, page_size), 2); // D(4) -> P0(2)
        assert_eq!(get_ptrmap_page_no_for_db_page(5, page_size), 2); // D(5) -> P0(2)
        assert_eq!(get_ptrmap_page_no_for_db_page(104, page_size), 2); // D(104) -> P0(2)

        assert_eq!(get_ptrmap_page_no_for_db_page(105, page_size), 105); // Page 105 is a pointer map page.

        // Test pages mapped by P1 (page 6)
        assert_eq!(get_ptrmap_page_no_for_db_page(106, page_size), 105); // D(106) -> P1(105)
        assert_eq!(get_ptrmap_page_no_for_db_page(107, page_size), 105); // D(107) -> P1(105)
        assert_eq!(get_ptrmap_page_no_for_db_page(108, page_size), 105); // D(108) -> P1(105)

        assert_eq!(get_ptrmap_page_no_for_db_page(208, page_size), 208); // Page 208 is a pointer map page.
    }

    #[test]
    fn test_get_ptrmap_offset() {
        let page_size = MIN_PAGE_SIZE as usize; //  Maps 103 data pages

        assert_eq!(get_ptrmap_offset_in_page(3, 2, page_size).unwrap(), 0);
        assert_eq!(
            get_ptrmap_offset_in_page(4, 2, page_size).unwrap(),
            1 * PTRMAP_ENTRY_SIZE
        );
        assert_eq!(
            get_ptrmap_offset_in_page(5, 2, page_size).unwrap(),
            2 * PTRMAP_ENTRY_SIZE
        );

        //  P1 (page 105) maps D(106)...D(207)
        // D(106) is index 0 on P1. Offset 0.
        // D(107) is index 1 on P1. Offset 5.
        // D(108) is index 2 on P1. Offset 10.
        assert_eq!(get_ptrmap_offset_in_page(106, 105, page_size).unwrap(), 0);
        assert_eq!(
            get_ptrmap_offset_in_page(107, 105, page_size).unwrap(),
            1 * PTRMAP_ENTRY_SIZE
        );
        assert_eq!(
            get_ptrmap_offset_in_page(108, 105, page_size).unwrap(),
            2 * PTRMAP_ENTRY_SIZE
        );
    }
}
