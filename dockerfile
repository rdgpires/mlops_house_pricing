# ---------- Base with micromamba (fast, small, conda-compatible) ----------
FROM mambaorg/micromamba:1.5.8

WORKDIR /app

# Build the conda env from your YAML (cached layer)
COPY conda_environment.yml /tmp/conda_environment.yml
RUN micromamba create -y -n housing -f /tmp/conda_environment.yml && \
    micromamba clean --all --yes

# Bring in the entrypoint script (kept in your repo)
COPY --chown=mambauser:mambauser --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh

# Expose both inference ports
EXPOSE 8000 8001

# Optional volume hint (you'll -v mount your project)
VOLUME ["/app"]

ENTRYPOINT ["/app/entrypoint.sh"]