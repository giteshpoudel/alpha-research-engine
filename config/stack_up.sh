#!/bin/zsh
# Ensure the Docker stack is up after a reboot/login, before scheduled jobs run.
# Installed OUTSIDE ~/Documents (launchd's /bin/zsh cannot read scripts there).
# Uses `docker start` (no compose-file read) so it needs no repo access.
# Idempotent: safe to run repeatedly.
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

if ! colima status >/dev/null 2>&1; then
    echo "$(date '+%F %T') starting colima"
    colima start
fi

# wait for the docker daemon (up to ~2 min)
for _ in $(seq 1 60); do
    docker info >/dev/null 2>&1 && break
    sleep 2
done

if docker info >/dev/null 2>&1; then
    echo "$(date '+%F %T') starting containers"
    docker start alpha-clickhouse alpha-qdrant 2>/dev/null || true
else
    echo "$(date '+%F %T') docker daemon not reachable"
    exit 1
fi
