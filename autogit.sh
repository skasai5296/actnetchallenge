#!/bin/bash

while [ : ]
do
	git add -A
	git commit -m "automatic daily commit"
	git push -u origin dev
	sleep 1m
done
