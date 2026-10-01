#!/bin/zsh
# Ensure Colima + the Docker stack are healthy. Runs every 5 min via launchd.
# Installed OUTSIDE ~/Documents (launchd's /bin/zsh cannot read scripts there).
# Idempotent and self-healing: Colima can report "running" while its Docker
# daemon is actually dead (the recurring dockerd-wedge), so liveness is judged
# by `docker info`, and a wedged daemon triggers a forced `colima restart`.
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

log() { echo "$(date '+%F %T') $*"; }
docker_ok() { docker info >/dev/null 2>&1; }
wait_docker() {
    for _ in $(seq 1 60); do docker_ok && return 0; sleep 2; done
    return 1
}

if ! colima status >/dev/null 2>&1; then
    log "colima not running -> starting"
    colima start
fi

if ! wait_docker; then
    log "docker daemon unreachable -> forcing colima restart"
    colima restart
    if ! wait_docker; then
        log "docker still unreachable after restart; aborting this cycle"
        exit 1
    fi
    log "docker recovered"
fi

log "ensuring containers are up"
docker start alpha-clickhouse alpha-qdrant 2>/dev/null || true
