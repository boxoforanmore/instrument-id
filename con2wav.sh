#!/bin/bash

echo
echo "Converting tune_type data from mp3 to more wav for preprocessing..."
echo

cd data/instruments

for fname in *
do
  echo "In $fname..."
  if [ -d "$fname" ]
  then
    # if fname is a directory, print dir name and cd into
    echo "Converting files to wav format in file: $fname"
    cd "$fname"
  
    # Format for loop from https://stackoverflow.com/questions/18621041/how-to-batch-convert-sph-files-to-wav-with-sox
    for f in *.mp3 
    do  
      sox -t mp3 "$f" -b 16 -t wav "${f%.*}.wav"
    done
    cd ../ 
  fi  
done

cd ../
