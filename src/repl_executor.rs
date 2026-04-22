use pyo3::prelude::*;

const BOOTSTRAP: &str = include_str!("repl_bootstrap.py");

pub struct ReplResult {
    pub success:        bool,
    pub output:         String,
    pub error:          String,
    pub namespace_json: String,
    pub trajectory_json: String,
}

fn find_soul_socket() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let dir = std::path::Path::new(&runtime_dir).join("chitta");
    std::fs::read_dir(&dir).ok()?.flatten().find(|e| {
        let n = e.file_name();
        let n = n.to_string_lossy();
        n.starts_with("chitta-") && n.ends_with(".sock")
    }).map(|e| e.path().to_string_lossy().into_owned())
}

pub fn repl_execute(
    code: &str,
    initial_namespace_json: Option<&str>,
    socket_path: &str,
    max_output: usize,
) -> ReplResult {
    pyo3::prepare_freethreaded_python();

    let sp = if socket_path.is_empty() {
        find_soul_socket().unwrap_or_default()
    } else {
        socket_path.to_string()
    };

    if sp.is_empty() {
        return ReplResult {
            success: false,
            output: String::new(),
            error: "no chitta socket found".to_string(),
            namespace_json: "{}".to_string(),
            trajectory_json: "[]".to_string(),
        };
    }

    let result = Python::with_gil(|py| -> PyResult<ReplResult> {
        let m = PyModule::from_code_bound(py, BOOTSTRAP, "repl_bootstrap.py", "repl_bootstrap")?;
        let func = m.getattr("repl_execute_main")?;
        let raw: String = func
            .call1((code, initial_namespace_json.unwrap_or(""), sp.as_str(), max_output as i64))?
            .extract()?;
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_default();
        Ok(ReplResult {
            success:         v["success"].as_bool().unwrap_or(false),
            output:          v["output"].as_str().unwrap_or("").to_string(),
            error:           v["error"].as_str().unwrap_or("").to_string(),
            namespace_json:  v["namespace_json"].as_str().unwrap_or("{}").to_string(),
            trajectory_json: v["trajectory"].to_string(),
        })
    });

    result.unwrap_or_else(|e| ReplResult {
        success:         false,
        output:          String::new(),
        error:           format!("pyo3: {e}"),
        namespace_json:  "{}".to_string(),
        trajectory_json: "[]".to_string(),
    })
}
