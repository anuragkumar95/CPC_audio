#!/bin/bash
# /pio/gluster/data/buckeye
# /pio/scratch/1/i325922/data/BUCKEYE/TRAIN
cd /pio/gluster/data/buckeye

for i in {1..40}
do
    if ((i < 10))
    then 

        wget "https://buckeyecorpus.osu.edu/speechfiles/s0$i.zip"
        unzip "s0$i.zip"                                           
        rm    "s0$i.zip"    
        for filename in /pio/gluster/data/buckeye/s0$i/*.zip
        do
            if [ "$filename" = "/pio/gluster/data/buckeye/s0$i/s0$i.zip" ]
            then 
                rm "$filename"
            else 
                utterance=$(basename -- "$filename")
                utterance="${utterance%.*}"
                mkdir "./s0$i/$utterance"
                unzip "$filename" -d "./s0$i/$utterance"
                mv "./s0$i/$utterance/$utterance.wav" "./s0$i/$utterance/s0$i-$utterance.wav"
                rm  "$filename"
            fi
        done

    cd /pio/gluster/data/buckeye

    else

        wget "https://buckeyecorpus.osu.edu/speechfiles/s$i.zip"
        unzip "s$i.zip"                                             
        rm    "s$i.zip"               
        for filename in /pio/gluster/data/buckeye/s$i/*.zip
        do
            if [ "$filename" = "/pio/gluster/data/buckeye/s$i/s$i.zip" ]
            then 
                rm "$filename"
            else
                utterance=$(basename -- "$filename")
                utterance="${utterance%.*}"
                mkdir "./s$i/$utterance"
                unzip "$filename" -d "./s$i/$utterance"
                mv "./s$i/$utterance/$utterance.wav" "./s$i/$utterance/s$i-$utterance.wav"
                rm  "$filename"
            fi
        done

    cd /pio/gluster/data/buckeye

    fi   
done
