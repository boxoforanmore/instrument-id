#!/bin/bash

cd data/instruments/

echo "Collecting 30 second snippets from various tunes"

declare -i total=0

for fname in *
do

  echo "Looking at $fname"
  if [ -d "$fname" ]
  then

    echo "Trimming files in $fname to be 30 seconds long (20s-50s)..."
    cd "$fname"
    declare -i number=0

    for f in *.mp3
    do
      echo "Trimming file number: $number..."
      echo "Filename: $f"

      sox "$f" "$fname$number.mp3" trim 20 30
      (( ++number ))
      (( ++total ))
    done
    cd ..

    echo
    echo 
    echo "Trimmed $number items from $fname"
    echo
    echo "Total trimmed so far => $total"

  fi
done

cd ../../
