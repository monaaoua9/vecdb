FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release --bin vecdb-server

FROM debian:bookworm-slim
COPY --from=build /src/target/release/vecdb-server /usr/local/bin/vecdb-server
VOLUME /data
EXPOSE 50051
ENTRYPOINT ["vecdb-server", "--listen", "0.0.0.0:50051", "--data-dir", "/data"]
