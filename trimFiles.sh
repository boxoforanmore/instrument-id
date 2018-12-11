#!/bin/bash

for fname in *
do

  echo "Looking at $fname"
  if [-d "$fname"]
  then

    echo "Trimming files in $fname to be from 30 seconds long (20s-50s)..."
    cd "$fname"
    number = 0

    for f in *.mp3
    do
      new_f = "$fname$number"
      sox "$f" "$fname$number.mp3" trim 20 50
      (( ++number ))
    done

  fi

done
