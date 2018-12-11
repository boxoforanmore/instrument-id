#!/bin/bash

for fname in *
do

  echo "Looking at $fname"
  if [-d "$fname"]
  then

    echo "Trimming files in $fname to be from 30 seconds long (20s-50s)..."
    cd "$fname"

    for f in *.mp3
    do
      sox "$f" "$f"2.mp3 trim 20 50
    done

  fi

done
