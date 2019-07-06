#!/bin/bash
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker volume rm openstreetmap-data
docker volume create openstreetmap-data -d local -o o=bind -o device=/media/yonatanz/yz/Research/coord2vec/open-streetmap-data
docker run -e THREADS=24 -v /media/yonatanz/yz/Research/coord2vec/osm/israel-and-palestine-latest.osm.pbf:/data.osm.pbf -v openstreetmap-data:/var/lib/postgresql/10/main overv/openstreetmap-tile-server import
docker run -it overv/openstreetmap-tile-server
#docker run -p 8080:80 -e THREADS=24 -v openstreetmap-data:/var/lib/postgresql/10/main -d overv/openstreetmap-tile-server run
