#![cfg(feature = "tls")]

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use tonic::transport::{Certificate, Channel, ClientTlsConfig};
use vecdb::server::pb::{self, vector_db_client::VectorDbClient};
use vecdb::server::{self, ServeOptions, TlsFiles};
use vecdb::{Collections, Config, DurableIndex, Index, Metric};

// self-signed for localhost / 127.0.0.1, only ever used by this test
const CERT: &[u8] = include_bytes!("data/localhost.crt");
const KEY: &[u8] = include_bytes!("data/localhost.key");

fn free_addr() -> std::net::SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap()
}

#[tokio::test]
async fn serves_over_tls_and_rejects_plaintext() {
    let index = Arc::new(DurableIndex::in_memory(Index::new(Config::new(2, Metric::L2)).unwrap()));
    let cols = Arc::new(Collections::new(index));
    let addr = free_addr();
    let (stop, rx) = oneshot::channel::<()>();
    let opts = ServeOptions {
        tls: Some(TlsFiles { cert_pem: CERT.to_vec(), key_pem: KEY.to_vec() }),
        ..Default::default()
    };
    let srv = tokio::spawn(server::serve_with(cols, addr, Duration::ZERO, opts, async {
        let _ = rx.await;
    }));

    let tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(CERT)).domain_name("localhost");
    let mut secure = None;
    let mut last = String::new();
    for _ in 0..20 {
        if srv.is_finished() {
            panic!("server stopped: {:?}", srv.await);
        }
        let ch = Channel::from_shared(format!("https://localhost:{}", addr.port()))
            .unwrap()
            .tls_config(tls.clone())
            .unwrap()
            .connect()
            .await;
        match ch {
            Ok(ch) => {
                secure = Some(VectorDbClient::new(ch));
                break;
            }
            Err(e) => last = format!("{e:?}"),
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let mut c = secure.unwrap_or_else(|| panic!("tls handshake never succeeded: {last}"));
    c.insert(pb::InsertRequest { id: 1, vector: vec![1.0, 2.0], payload: None }).await.unwrap();
    assert_eq!(c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner().vector, vec![1.0, 2.0]);

    // a client that doesn't speak TLS never gets an answer (the server waits for a handshake)
    let plain = tokio::time::timeout(Duration::from_secs(3), async {
        let mut p = VectorDbClient::connect(format!("http://127.0.0.1:{}", addr.port())).await.ok()?;
        p.get(pb::GetRequest { id: 1 }).await.ok()
    })
    .await;
    assert!(matches!(plain, Err(_) | Ok(None)), "plaintext request succeeded");

    let _ = stop.send(());
}
