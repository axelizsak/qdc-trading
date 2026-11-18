#!/usr/bin/env bash
# Installs the automatic morning (scripts/morning.sh) as a launchd job of the current user: Monday to Friday at 7:30,
# and at each login (a Mac switched on at 9:00 runs it then; one asleep at 7:30 runs it when it wakes). The script
# itself does nothing on weekends, and a second run the same day costs nothing.
#
# macOS keeps background jobs out of the Desktop, Documents and Downloads folders: with the project there, give Full
# Disk Access to /bin/bash once (System Settings > Privacy & Security > Full Disk Access), or keep the project
# elsewhere.
#
# Usage:  scripts/install_morning.sh            (install, or update)
#         scripts/install_morning.sh --remove   (uninstall)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
label="com.qdc-trading.morning"
plist="$HOME/Library/LaunchAgents/$label.plist"
log="$HOME/Library/Logs/qdc-morning.log"
domain="gui/$(id -u)"

launchctl bootout "$domain" "$plist" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then
    rm -f "$plist"
    echo "morning run removed"
    exit 0
fi

days=""
for day in 1 2 3 4 5; do
    days="$days<dict><key>Weekday</key><integer>$day</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>"
done
mkdir -p "$(dirname "$plist")" "$(dirname "$log")"
cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$root/scripts/morning.sh</string></array>
  <key>StartCalendarInterval</key><array>$days</array>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$log</string>
  <key>StandardErrorPath</key><string>$log</string>
</dict>
</plist>
EOF
launchctl bootstrap "$domain" "$plist"
echo "morning run installed: Monday to Friday at 7:30, output in $log"
echo "run it now:  launchctl kickstart $domain/$label"
