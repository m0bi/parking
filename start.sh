#!/bin/sh

mkdir data
cd ./data
wget https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv
cd ..
FLASK_APP=server.py flask run