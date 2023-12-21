#!/bin/bash
# We assume logs are saved under slurm/logs, this retrieves the last created logs
LOG_FILE_PATH=$(find /fsx/loubna/projects/tgi-swarm/slurm/logs -type f -name "*.out" -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2)

echo "Latest created log file is $LOG_FILE_PATH"

# Extracting the port number
PORT=$(grep -oP ' port: \K\d+' "$LOG_FILE_PATH")
if [ -z "$port" ]; then
    echo "Port not found in log file."
else
    echo "Port: $port"
fi


# Extracting the hostname
HOSTNAME=$(grep -m 1 "hostname:" "$LOG_FILE_PATH" | awk -F '\"' '{print $4}')
if [ -z "$HOSTNAME" ]; then
    echo "Hostname not found in log file."
else
    echo "Hostname: $HOSTNAME"
fi

ADDRESS="http://$HOSTNAME:$PORT"

echo "Saving address $ADDRESS in hosts.txt"
rm -f $PWD/hosts.txt
touch $PWD/hosts.txt
echo $ADDRESS >> $PWD/hosts.txt

echo "Testing the endpoint works. Output:"
if curl -m 10 $ADDRESS/generate \
    -X POST \
    -d '{"inputs":"What is Life?","parameters":{"max_new_tokens":10}}' \
    -H 'Content-Type: application/json'; then
    echo -e "\nThe endpoint works 🎉!"
else
    echo "curl command failed."
    echo "\nDisplaying the last four lines of the log file:"
    tail -n 4 "$LOG_FILE_PATH"
fi
exit
