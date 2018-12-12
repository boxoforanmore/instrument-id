#!/bin/bash

echo "Enter the file you want to trim: "

read fname

echo "Enter the time to start trimmming: "

read begin

echo "What instrument is being used in this file?"

read instrument


echo "Trimming file $fname for a 30 second clip starting from $begin seconds"

sox "$fname" "$instrument$begin.mp3" trim $begin 30
