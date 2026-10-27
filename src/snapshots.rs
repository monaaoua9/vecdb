use std::path::{Path, PathBuf};

// Snapshots are written as snapshot-<generation>.vdb and never overwritten, so
// a running server (which keeps the previous one mapped) can always write the next.

fn generation(p: &Path) -> Option<u64> {
    let name = p.file_name()?.to_str()?;
    name.strip_prefix("snapshot-")?.strip_suffix(".vdb")?.parse().ok()
}

fn all(dir: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut v: Vec<_> =
        rd.filter_map(|e| e.ok()).filter_map(|e| Some((generation(&e.path())?, e.path()))).collect();
    v.sort();
    v
}

pub fn latest(dir: &Path) -> Option<(u64, PathBuf)> {
    all(dir).pop()
}

pub fn path_for(dir: &Path, gen_no: u64) -> PathBuf {
    dir.join(format!("snapshot-{gen_no:012}.vdb"))
}

/// keeps the newest `keep` files; a file still mapped on Windows just fails to delete and is retried next time
pub fn prune(dir: &Path, keep: usize) {
    let files = all(dir);
    let cut = files.len().saturating_sub(keep);
    for (_, p) in &files[..cut] {
        if let Err(e) = std::fs::remove_file(p) {
            tracing::debug!(path = %p.display(), error = %e, "old snapshot not removed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_highest_generation() {
        let d = tempfile::tempdir().unwrap();
        for g in [3u64, 11, 7] {
            std::fs::write(path_for(d.path(), g), b"x").unwrap();
        }
        std::fs::write(d.path().join("notes.txt"), b"x").unwrap();
        assert_eq!(latest(d.path()).unwrap().0, 11);
        prune(d.path(), 2);
        assert_eq!(all(d.path()).len(), 2);
        assert_eq!(latest(d.path()).unwrap().0, 11);
    }
}
