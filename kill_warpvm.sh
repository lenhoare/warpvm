#!/usr/bin/env bash
# Stop only this user's processes whose exact executable name is "warpvm".
set -u

readonly WARPVM_USER_ID="$(id -u)"

is_warpvm_process() {
    local pid="$1"
    local name owner

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/$pid/comm" ]] || return 1
    IFS= read -r name < "/proc/$pid/comm" || return 1
    [[ "$name" == "warpvm" ]] || return 1
    owner="$(stat -c '%u' "/proc/$pid" 2>/dev/null)" || return 1
    [[ "$owner" == "$WARPVM_USER_ID" ]]
}

mapfile -t candidates < <(pgrep -u "$WARPVM_USER_ID" -x warpvm 2>/dev/null || true)
targets=()
for pid in "${candidates[@]}"; do
    if is_warpvm_process "$pid"; then
        targets+=("$pid")
    fi
done

if ((${#targets[@]} == 0)); then
    echo "No warpvm processes are running for this user."
    exit 0
fi

echo "Stopping warpvm processes: ${targets[*]}"
for pid in "${targets[@]}"; do
    is_warpvm_process "$pid" && kill -TERM "$pid" 2>/dev/null || true
done

# Give viewers and persistent CUDA runtimes two seconds to shut down cleanly.
for _ in {1..20}; do
    survivors=()
    for pid in "${targets[@]}"; do
        if is_warpvm_process "$pid" && kill -0 "$pid" 2>/dev/null; then
            survivors+=("$pid")
        fi
    done
    ((${#survivors[@]} == 0)) && break
    sleep 0.1
done

if ((${#survivors[@]} != 0)); then
    echo "Force-stopping remaining warpvm processes: ${survivors[*]}"
    for pid in "${survivors[@]}"; do
        is_warpvm_process "$pid" && kill -KILL "$pid" 2>/dev/null || true
    done
fi

echo "All warpvm processes for this user have been stopped."
