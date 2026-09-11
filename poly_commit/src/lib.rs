#![allow(clippy::manual_div_ceil)]

mod traits;
pub use traits::{BatchOpeningPCS, PolynomialCommitmentScheme};

pub const PCS_SOUNDNESS_BITS: usize = 128;

mod utils;
pub use utils::expander_pcs_init_testing_only;

pub mod raw;
pub use raw::RawExpanderGKR;

pub mod orion;
pub use orion::*;

pub mod hyrax;
pub use hyrax::*;

pub mod kzg;
pub use kzg::*;

pub mod batching;

/// The GPU-resident Orion open path references CUDA kernels that are only built on Linux. On other
/// platforms these stubs satisfy the linker. They are unreachable there: the GPU commit path that
/// registers a tree is compiled out, so `GPU_TREE_REGISTRY` never has an entry.
#[cfg(not(target_os = "linux"))]
mod gpu_stubs {
    #[no_mangle]
    pub extern "C" fn gpu_tree_free(_tree_id: i32) {
        unreachable!("GPU trees are not available on this platform")
    }

    #[no_mangle]
    pub extern "C" fn gpu_tree_get_ptrs(
        _id: i32, _leaves: *mut *mut u32, _lh: *mut *mut u8, _nd: *mut *mut u8, _n: *mut u32,
        _poly: *mut *mut u32, _cl: *mut u32, _ml: *mut u32,
    ) {
        unreachable!("GPU trees are not available on this platform")
    }

    #[no_mangle]
    pub extern "C" fn gpu_pcs_open_with_device_data(
        _d_packed_evals: *const u32, _h_eq_coeffs: *const u32, _packed_rows: u32, _msg_len: u32,
        _h_eval_row: *mut u32, _n_proximity: u32, _h_rand_coeffs: *const u32, _h_prox_rows: *mut u32,
        _tree_id: i32, _h_query_indices: *const u32, _n_queries: u32, _leaves_per_query: u32,
        _h_query_leaves: *mut u8, _h_query_path_nodes: *mut u8, _out_tree_depth: *mut u32,
    ) {
        unreachable!("GPU trees are not available on this platform")
    }
}
