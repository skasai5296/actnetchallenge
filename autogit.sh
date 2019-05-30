#!/bin/bash

while [ : ]
do
	git add -A
	git commit -m "automatic commit"
	git push -u origin dev
	sleep 1m
done
