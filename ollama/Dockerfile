FROM dustynv/ollama:0.6.8-r36.4-cu126-22.04

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Add debugging
ENV OLLAMA_DEBUG=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]