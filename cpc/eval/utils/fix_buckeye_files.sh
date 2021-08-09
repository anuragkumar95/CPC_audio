#!/bin/bash

for dir in /pio/scratch/1/i325922/data/BUCKEYE/clean/all-clean/*
    do

    for file in $dir/*
    do

    mv $file/* $dir
    rm -r "${file%%.*}"

    #mkdir "${file%%.*}"
    #mv $file "${file%%.*}"

    done

    done