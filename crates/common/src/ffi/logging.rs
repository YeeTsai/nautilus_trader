// -------------------------------------------------------------------------------------------------
//  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
//  https://nautechsystems.io
//
//  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
//  You may not use this file except in compliance with the License.
//  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// -------------------------------------------------------------------------------------------------

use std::{
    ffi::c_char,
    ops::{Deref, DerefMut},
};

use ahash::AHashMap;
use nautilus_core::{
    UUID4,
    ffi::{
        parsing::{optional_bytes_to_json, u8_as_bool},
        string::{cstr_as_str, cstr_to_ustr, optional_cstr_to_str},
    },
};
use nautilus_model::identifiers::TraderId;

use crate::{
    enums::{LogColor, LogLevel},
    logging::{
        headers, init_logging,
        logger::{self, LogGuard, LoggerConfig},
        map_log_level_to_filter, parse_component_levels,
        writer::FileWriterConfig,
    },
};

/// C compatible Foreign Function Interface (FFI) for an underlying [`LogGuard`].
///
/// This struct wraps `LogGuard` in a way that makes it compatible with C function
/// calls, enabling interaction with `LogGuard` in a C environment.
///
/// It implements the `Deref` trait, allowing instances of `LogGuard_API` to be
/// dereferenced to `LogGuard`, providing access to `LogGuard`'s methods without
/// having to manually access the underlying `LogGuard` instance.
///
/// The inner pointer is nullable: `logging_init` returns a null guard when the
/// logging subsystem cannot be re-initialized (see its docs). `Option<Box<T>>`
/// is guaranteed to use the null-pointer representation, so the C layout is
/// still a single `struct LogGuard *`.
#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct LogGuard_API(Option<Box<LogGuard>>);

impl Deref for LogGuard_API {
    type Target = LogGuard;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("LogGuard_API is null")
    }
}

impl DerefMut for LogGuard_API {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("LogGuard_API is null")
    }
}

/// Initializes logging.
///
/// Logging should be used for Python and sync Rust logic which is most of
/// the components in the [nautilus_trader](https://pypi.org/project/nautilus_trader) package.
/// Logging can be configured to filter components and write up to a specific level only
/// by passing a configuration using the `NAUTILUS_LOG` environment variable.
///
/// # Safety
///
/// Should only be called once during an application's run, ideally at the
/// beginning of the run.
///
/// This function assumes:
/// - `directory_ptr` is either NULL or a valid C string pointer.
/// - `file_name_ptr` is either NULL or a valid C string pointer.
/// - `file_format_ptr` is either NULL or a valid C string pointer.
/// - `component_level_ptr` is either NULL or a valid C string pointer.
///
/// # Panics
///
/// Panics if the component log levels cannot be parsed, or if initializing the
/// Rust logger fails for any reason other than re-initialization.
///
/// Returns a null guard (rather than panicking) when the logging subsystem
/// cannot be re-initialized after a previous `LogGuard` was dropped: the `log`
/// crate's global logger can only be set once per process, so this is a normal
/// outcome for a second kernel in the same process, not a fault.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn logging_init(
    trader_id: TraderId,
    instance_id: UUID4,
    level_stdout: LogLevel,
    level_file: LogLevel,
    directory_ptr: *const c_char,
    file_name_ptr: *const c_char,
    file_format_ptr: *const c_char,
    component_levels_ptr: *const c_char,
    is_colored: u8,
    is_bypassed: u8,
    print_config: u8,
    log_components_only: u8,
    max_file_size: u64,
    max_backup_count: u32,
) -> LogGuard_API {
    let level_stdout = map_log_level_to_filter(level_stdout);
    let level_file = map_log_level_to_filter(level_file);

    let component_levels_json = unsafe { optional_bytes_to_json(component_levels_ptr) };
    let component_levels = parse_component_levels(component_levels_json)
        .expect("Failed to parse component log levels");

    let config = LoggerConfig::new(
        level_stdout,
        level_file,
        component_levels,
        AHashMap::new(), // module_level - not exposed to FFI
        u8_as_bool(log_components_only),
        u8_as_bool(is_colored),
        u8_as_bool(print_config),
        false, // use_tracing - not exposed to FFI
        u8_as_bool(is_bypassed),
        None,  // file_config - passed separately to init_logging
        false, // clear_log_file
    );

    // Configure file rotation if max_file_size > 0
    let file_rotate = if max_file_size > 0 {
        Some((max_file_size, max_backup_count))
    } else {
        None
    };

    let directory = unsafe { optional_cstr_to_str(directory_ptr).map(ToString::to_string) };
    let file_name = unsafe { optional_cstr_to_str(file_name_ptr).map(ToString::to_string) };
    let file_format = unsafe { optional_cstr_to_str(file_format_ptr).map(ToString::to_string) };

    let file_config = FileWriterConfig::new(directory, file_name, file_format, file_rotate);

    if u8_as_bool(is_bypassed) {
        logging_set_bypass();
    }

    match init_logging(trader_id, instance_id, config, file_config) {
        Ok(guard) => LogGuard_API(Some(Box::new(guard))),
        // Re-initialization only: the `log` crate refuses a second
        // `set_boxed_logger`. Hand back a null guard and let the caller decide;
        // the Cython layer turns it into `LoggingReinitError`.
        Err(e) if e.downcast_ref::<log::SetLoggerError>().is_some() => LogGuard_API(None),
        // Any other failure (e.g. the log thread failing to spawn) is a real
        // fault on first initialization and stays as loud as it is today.
        Err(e) => panic!("Failed to initialize logging: {e:?}"),
    }
}

/// Creates a new log event.
///
/// # Safety
///
/// This function assumes:
/// - `component_ptr` is a valid C string pointer.
/// - `message_ptr` is a valid C string pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn logger_log(
    level: LogLevel,
    color: LogColor,
    component_ptr: *const c_char,
    message_ptr: *const c_char,
) {
    let component = unsafe { cstr_to_ustr(component_ptr) };
    let message = unsafe { cstr_as_str(message_ptr) };

    logger::log(level, color, component, message);
}

/// Logs the Nautilus system header.
///
/// # Safety
///
/// This function assumes:
/// - `machine_id_ptr` is a valid C string pointer.
/// - `component_ptr` is a valid C string pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn logging_log_header(
    trader_id: TraderId,
    machine_id_ptr: *const c_char,
    instance_id: UUID4,
    component_ptr: *const c_char,
) {
    let component = unsafe { cstr_to_ustr(component_ptr) };
    let machine_id = unsafe { cstr_as_str(machine_id_ptr) };
    headers::log_header(trader_id, machine_id, instance_id, component);
}

/// Logs system information.
///
/// # Safety
///
/// Assumes `component_ptr` is a valid C string pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn logging_log_sysinfo(component_ptr: *const c_char) {
    let component = unsafe { cstr_to_ustr(component_ptr) };
    headers::log_sysinfo(component);
}

/// Flushes global logger buffers of any records.
#[unsafe(no_mangle)]
pub extern "C" fn logger_flush() {
    log::logger().flush();
}

/// Flushes global logger buffers of any records and then drops the logger.
#[unsafe(no_mangle)]
pub extern "C" fn logger_drop(log_guard: LogGuard_API) {
    drop(log_guard);
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_is_initialized() -> u8 {
    u8::from(crate::logging::logging_is_initialized())
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_set_bypass() {
    crate::logging::logging_set_bypass();
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_shutdown() {
    crate::logging::logging_shutdown();
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_is_colored() -> u8 {
    u8::from(crate::logging::logging_is_colored())
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_clock_set_realtime_mode() {
    crate::logging::logging_clock_set_realtime_mode();
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_clock_set_static_mode() {
    crate::logging::logging_clock_set_static_mode();
}

#[unsafe(no_mangle)]
pub extern "C" fn logging_clock_set_static_time(time_ns: u64) {
    crate::logging::logging_clock_set_static_time(time_ns);
}
