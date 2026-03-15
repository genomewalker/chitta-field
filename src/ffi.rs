use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use crate::field::ChittaField;

/// Opaque handle to a ChittaField instance
pub struct CfHandle {
    field: ChittaField,
    last_error: Option<CString>,
}

#[no_mangle]
pub extern "C" fn cf_open(
    data_dir: *const c_char,
    lock_dir: *const c_char,
) -> *mut CfHandle {
    let data_dir = unsafe {
        match CStr::from_ptr(data_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let lock_dir = unsafe {
        match CStr::from_ptr(lock_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    match ChittaField::open(data_dir, lock_dir) {
        Ok(field) => Box::into_raw(Box::new(CfHandle { field, last_error: None })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn cf_close(handle: *mut CfHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[no_mangle]
pub extern "C" fn cf_last_error(handle: *mut CfHandle) -> *const c_char {
    if handle.is_null() {
        return std::ptr::null();
    }
    unsafe {
        (*handle).last_error.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null())
    }
}
