FROM python:3.10-slim-bullseye

EXPOSE 8080

RUN pip install --no-cache-dir gunicorn

WORKDIR /app

RUN apt update
RUN apt upgrade -y
RUN apt install -y git

# Add known hosts for remote repositories: Needed for private repositories in requirements
RUN mkdir -p -m 0600 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts

COPY ../pyproject.toml .

# Use SSH mount to access to external SSH agent during dependency insallation.
# Install dependencies from pyproject.toml then uninstall the dummy package (src is empty at this stage)
RUN --mount=type=ssh \
    pip install --no-cache-dir --upgrade . && \
    pip uninstall -y dev_ops_demo

COPY ./src .

RUN chmod +x ./gunicorn_starter.sh

RUN useradd appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["./gunicorn_starter.sh"]

# If ther's the need to keep the container running for debug purposes: ENTRYPOINT ["tail", "-f", "/dev/null"]