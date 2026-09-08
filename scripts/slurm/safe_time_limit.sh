#!/usr/bin/env bash
# Print a --time value that SLURM will actually schedule: capped so the job ends BEFORE the
# next maintenance reservation.
#
# WHY THIS EXISTS. SLURM will not start a job whose wall clock runs past a reservation, and
# it says so only as `Reason=ReqNodeNotAvail,_Reserved_for_maintenance` with StartTime pushed
# to the far side. It has bitten this project three times -- 2026-07-30, 2026-08-30 and
# 2026-09-03, the last costing 15 jobs a 5.5-day delay -- because train_pusht_search.sbatch
# asks for 10 days and nobody checks the reservation calendar before submitting.
#
#   TIME_LIMIT=$(bash scripts/slurm/safe_time_limit.sh)          # cap at 3 days
#   TIME_LIMIT=$(bash scripts/slurm/safe_time_limit.sh 5-00:00:00)
set -uo pipefail
CAP="${1:-3-00:00:00}"
MARGIN_SEC="${MARGIN_SEC:-3600}"      # finish an hour clear of the window

to_sec() {  # D-HH:MM:SS | HH:MM:SS -> seconds
    local t="$1" d=0
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}";; esac
    IFS=: read -r h m s <<<"$t"
    echo $(( 10#$d*86400 + 10#${h:-0}*3600 + 10#${m:-0}*60 + 10#${s:-0} ))
}
fmt() { printf '%d-%02d:%02d:00\n' $(( $1/86400 )) $(( ($1%86400)/3600 )) $(( ($1%3600)/60 )); }

cap_sec=$(to_sec "$CAP")
now=$(date +%s)
soonest=""
while read -r st; do
    [ -z "$st" ] && continue
    s=$(date -d "$st" +%s 2>/dev/null) || continue
    [ "$s" -le "$now" ] && continue
    { [ -z "$soonest" ] || [ "$s" -lt "$soonest" ]; } && soonest=$s
done < <(scontrol show reservation 2>/dev/null | grep -oP 'StartTime=\K[0-9T:-]+')

if [ -z "$soonest" ]; then
    echo "$CAP"; exit 0
fi
avail=$(( soonest - now - MARGIN_SEC ))
if [ "$avail" -le 0 ]; then
    # Inside (or right up against) the window: ask for the cap and let it queue past it.
    echo "$CAP"; exit 0
fi
[ "$avail" -gt "$cap_sec" ] && avail=$cap_sec
fmt "$avail"
