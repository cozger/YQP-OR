#!/usr/bin/env bash
#
# Enrollment Worker Thread Diagnostic Verification Script
#
# This script checks if the EnrollmentWorkerThread.run() method is executing
# by examining the diagnostic log file written by the thread.
#

echo "================================================================================"
echo "Enrollment Worker Thread Diagnostic Check"
echo "================================================================================"
echo ""

DIAG_FILE="/tmp/enrollment_worker_diagnostic.log"

# Check if diagnostic file exists
if [ ! -f "$DIAG_FILE" ]; then
    echo "❌ DIAGNOSTIC FILE NOT FOUND: $DIAG_FILE"
    echo ""
    echo "This means the EnrollmentWorkerThread.run() method has NOT executed yet."
    echo ""
    echo "Possible causes:"
    echo "  1. Application hasn't been launched yet"
    echo "  2. Thread creation failed during initialization"
    echo "  3. Thread.start() was called but run() method never executed"
    echo "  4. Thread died immediately due to exception or stop_event"
    echo ""
    echo "Next steps:"
    echo "  1. Launch the application: ./run.sh"
    echo "  2. Wait for GUI to appear"
    echo "  3. Run this script again: ./check_enrollment_worker.sh"
    echo ""
    exit 1
fi

# File exists - display information
echo "✅ DIAGNOSTIC FILE EXISTS: $DIAG_FILE"
echo ""

# Display file metadata
echo "File Information:"
echo "  Location: $DIAG_FILE"
ls -lh "$DIAG_FILE" | awk '{print "  Size: " $5 ", Modified: " $6 " " $7 " " $8}'
echo ""

# Display file contents
echo "File Contents:"
echo "--------------------------------------------------------------------------------"
cat "$DIAG_FILE"
echo "--------------------------------------------------------------------------------"
echo ""

# Check if file was recently updated (within last 60 seconds)
CURRENT_TIME=$(date +%s)
FILE_TIME=$(stat -c %Y "$DIAG_FILE" 2>/dev/null || stat -f %m "$DIAG_FILE" 2>/dev/null)
TIME_DIFF=$((CURRENT_TIME - FILE_TIME))

if [ $TIME_DIFF -lt 60 ]; then
    echo "✅ SUCCESS: run() method HAS EXECUTED! (File updated $TIME_DIFF seconds ago)"
    echo ""
    echo "The EnrollmentWorkerThread is running and processing enrollment tasks."
elif [ $TIME_DIFF -lt 300 ]; then
    echo "⚠️  WARNING: File was updated $TIME_DIFF seconds ago (5 min threshold)"
    echo ""
    echo "The thread may have executed but stopped, or the application is not running."
else
    echo "⚠️  WARNING: File is old (updated $TIME_DIFF seconds ago)"
    echo ""
    echo "This file may be from a previous run. Try:"
    echo "  1. Remove old file: rm $DIAG_FILE"
    echo "  2. Restart application: ./run.sh"
    echo "  3. Re-run this script: ./check_enrollment_worker.sh"
fi

# Check if task processing occurred
if grep -q "PROCESSING FIRST TASK" "$DIAG_FILE"; then
    echo ""
    echo "✅ TASK PROCESSING CONFIRMED: Thread has processed enrollment tasks!"
else
    echo ""
    echo "⚠️  No task processing detected yet. Thread is running but queue may be empty."
    echo "   This is normal if no faces have been detected yet."
fi

echo ""
echo "================================================================================"
