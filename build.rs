fn main() {
    #[cfg(feature = "server")]
    {
        println!("cargo:rerun-if-changed=proto/vectordb.proto");
        tonic_prost_build::compile_protos("proto/vectordb.proto").expect("failed to compile proto");
    }
}
