use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PredicateStatus {
    Unknown,  // never run
    Passed,   // last run exited 0
    Failed,   // last run exited non-zero
}

impl Default for PredicateStatus {
    fn default() -> Self { PredicateStatus::Unknown }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPredicate {
    pub predicate_id:    u64,
    pub memory_id:       u64,
    pub check_cmd:       String,
    pub status:          PredicateStatus,
    pub last_checked_ms: Option<i64>,
    pub last_output:     Option<String>,
    pub created_ms:      i64,
}

/// Epistemic status derived from the predicate set for a memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EpistemicStatus {
    Asserted, // no predicates attached
    Tested,   // at least one predicate passed, none failed
    Failed,   // at least one predicate failed
    Mixed,    // some passed, some failed
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredicateStore {
    pub predicates:        Vec<MemoryPredicate>,
    pub next_predicate_id: u64,
}

impl PredicateStore {
    /// Attach a new predicate to a memory (does not run it).
    pub fn attach(&mut self, memory_id: u64, check_cmd: String, now_ms: i64) -> u64 {
        let id = self.next_predicate_id;
        self.next_predicate_id += 1;
        self.predicates.push(MemoryPredicate {
            predicate_id:    id,
            memory_id,
            check_cmd,
            status:          PredicateStatus::Unknown,
            last_checked_ms: None,
            last_output:     None,
            created_ms:      now_ms,
        });
        id
    }

    pub fn for_memory(&self, memory_id: u64) -> Vec<&MemoryPredicate> {
        self.predicates.iter().filter(|p| p.memory_id == memory_id).collect()
    }

    /// Run all predicates for a memory. Returns (passed, failed) counts.
    /// Runs each check_cmd via sh -c with a 5s timeout.
    pub fn run(&mut self, memory_id: u64, now_ms: i64) -> (usize, usize) {
        let ids: Vec<u64> = self.predicates.iter()
            .filter(|p| p.memory_id == memory_id)
            .map(|p| p.predicate_id)
            .collect();

        let mut passed = 0usize;
        let mut failed = 0usize;

        for pred_id in ids {
            if let Some(p) = self.predicates.iter().find(|p| p.predicate_id == pred_id) {
                let cmd = p.check_cmd.clone();
                let result = run_cmd(&cmd);
                let (ok, out) = result;
                if let Some(p) = self.predicates.iter_mut().find(|p| p.predicate_id == pred_id) {
                    p.status = if ok { PredicateStatus::Passed } else { PredicateStatus::Failed };
                    p.last_checked_ms = Some(now_ms);
                    p.last_output = Some(out.chars().take(400).collect());
                }
                if ok { passed += 1; } else { failed += 1; }
            }
        }
        (passed, failed)
    }

    pub fn epistemic_status(&self, memory_id: u64) -> EpistemicStatus {
        let preds = self.for_memory(memory_id);
        if preds.is_empty() { return EpistemicStatus::Asserted; }
        let has_pass = preds.iter().any(|p| p.status == PredicateStatus::Passed);
        let has_fail = preds.iter().any(|p| p.status == PredicateStatus::Failed);
        match (has_pass, has_fail) {
            (true,  false) => EpistemicStatus::Tested,
            (false, true)  => EpistemicStatus::Failed,
            (true,  true)  => EpistemicStatus::Mixed,
            (false, false) => EpistemicStatus::Asserted, // all Unknown
        }
    }
}

/// Run a shell command with a 5s timeout. Returns (exit_ok, combined_output).
///
/// popen() forks and pipes output correctly even when the daemon has SIGCHLD
/// set to SIG_IGN (llama.cpp side-effect). However pclose() returns -1/ECHILD
/// when the child is auto-reaped under SIG_IGN — so we cannot use its return
/// value for the exit code.
///
/// Instead we append a sentinel `__EXIT__:<code>` to the piped output and
/// parse the exit code ourselves, making us completely independent of pclose().
fn run_cmd(cmd: &str) -> (bool, String) {
    use std::ffi::CString;

    // The '; printf ...' runs AFTER timeout completes, at the outer popen shell
    // level, capturing timeout's own exit code (0=pass, non-zero=fail/timeout).
    let full_cmd = format!(
        "/usr/bin/timeout 5 sh -c {}; printf '__EXIT__:%d\\n' $?",
        shell_escape(cmd)
    );
    let Ok(cstr) = CString::new(full_cmd) else {
        return (false, "invalid command bytes".to_string());
    };
    let Ok(mode) = CString::new("r") else { unreachable!() };

    let fp = unsafe { libc::popen(cstr.as_ptr(), mode.as_ptr()) };
    if fp.is_null() {
        let errno = unsafe { *libc::__errno_location() };
        return (false, format!("popen failed errno={errno}"));
    }

    let mut raw = Vec::<u8>::new();
    let mut buf = [0u8; 256];
    loop {
        let n = unsafe {
            libc::fread(buf.as_mut_ptr() as *mut libc::c_void, 1, buf.len(), fp)
        };
        if n == 0 { break; }
        raw.extend_from_slice(&buf[..n]);
    }
    unsafe { libc::pclose(fp) }; // return value unreliable under SIG_IGN — ignore

    let text = String::from_utf8_lossy(&raw);
    if let Some(pos) = text.rfind("__EXIT__:") {
        let code: i32 = text[pos + 9..].trim().parse().unwrap_or(1);
        let user_output = text[..pos].to_string();
        (code == 0, user_output.chars().take(400).collect())
    } else {
        // Sentinel absent — popen failed or timed out hard before printf ran.
        (false, text.chars().take(400).collect())
    }
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}
