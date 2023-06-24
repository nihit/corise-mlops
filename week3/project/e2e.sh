#!/bin/bash -e

while IFS= read -r line; do

  curl --location 'http://localhost/predict' \
    --header 'accept: application/json' \
    --header 'Content-Type: application/json' \
    --data "$line"
    echo ""

done < data/requests.json
