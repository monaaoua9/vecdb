use std::path::Path;
use std::sync::Arc;

use tonic::{Request, Status};

/// Accepted API keys, sent by clients as `authorization: Bearer <key>`.
#[derive(Clone)]
pub struct ApiKeys(Arc<Vec<Vec<u8>>>);

impl ApiKeys {
    pub fn new(keys: impl IntoIterator<Item = String>) -> Self {
        ApiKeys(Arc::new(keys.into_iter().map(String::into_bytes).collect()))
    }

    /// one key per line; blank lines and lines starting with `#` are skipped
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let keys: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_owned)
            .collect();
        if keys.is_empty() {
            return Err(std::io::Error::other("api key file has no keys"));
        }
        Ok(Self::new(keys))
    }

    pub fn check(&self, req: Request<()>) -> Result<Request<()>, Status> {
        let given = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| Status::unauthenticated("missing bearer token"))?;
        // look at every key so timing doesn't say which one was close
        let ok = self.0.iter().fold(false, |acc, k| acc | same(k, given.as_bytes()));
        if ok { Ok(req) } else { Err(Status::unauthenticated("invalid api key")) }
    }
}

fn same(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().min(b.len()) {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(token: Option<&str>) -> Request<()> {
        let mut r = Request::new(());
        if let Some(t) = token {
            r.metadata_mut().insert("authorization", t.parse().unwrap());
        }
        r
    }

    #[test]
    fn accepts_only_listed_keys() {
        let k = ApiKeys::new(["alpha".to_string(), "beta".to_string()]);
        assert!(k.check(req(Some("Bearer beta"))).is_ok());
        assert!(k.check(req(Some("Bearer alph"))).is_err());
        assert!(k.check(req(Some("Bearer alphaa"))).is_err());
        assert!(k.check(req(Some("alpha"))).is_err());
        assert!(k.check(req(None)).is_err());
    }
}
