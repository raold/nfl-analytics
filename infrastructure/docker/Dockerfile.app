FROM rocker/r-ver:4.3.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    libpq-dev libssl-dev libffi-dev pkg-config \
    libxml2-dev libcurl4-openssl-dev \
    postgresql-client \
    python3-pip python3-venv \
    wget gdebi-core \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip

# Install Quarto CLI (arm64 tarball)
ARG QUARTO_VERSION=1.6.35
RUN wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-arm64.tar.gz \
 && tar -xzf quarto-${QUARTO_VERSION}-linux-arm64.tar.gz -C /opt \
 && ln -sf /opt/quarto-${QUARTO_VERSION}/bin/quarto /usr/local/bin/quarto \
 && rm -f quarto-${QUARTO_VERSION}-linux-arm64.tar.gz

WORKDIR /workspace

# Pre-install Python deps using system python
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Seed R packages (build cache) – safe to ignore failures and re-run at runtime
# Optional: full R dependency bootstrap is run at runtime by scripts/dev_setup.sh
COPY setup_packages.R /opt/setup_packages.R

ENV POSTGRES_HOST=pg \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=devdb01 \
    POSTGRES_USER=dro \
    POSTGRES_PASSWORD=sicillionbillions

CMD ["bash", "-lc", "sleep infinity"]
