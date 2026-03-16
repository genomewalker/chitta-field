#!/usr/bin/env bash
# Emergency: remove /tmp/chitta_mig.duckdb which filled /tmp
rm -f /tmp/chitta_mig.duckdb
echo "removed, space now:" > /maps/projects/fernandezguerra/apps/repos/chitta-field/scripts/cleanup_result.txt
df -h /tmp >> /maps/projects/fernandezguerra/apps/repos/chitta-field/scripts/cleanup_result.txt 2>&1
