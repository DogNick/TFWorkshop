#!/bin/bash

BEGIN=$1
END=$2
FLAG=$3
if [ "$FLAG"x = "file"x ]
then
	ssh guest@10.143.53.234 "cd /search/odin/softlink/qq; ls qq* | sort -t \"-\" -k3n -k4.1,4.2n | \
								awk -F'-|_' -v b=$BEGIN -v e=$END 'BEGIN{gsub(/\-/, \"\", b); gsub(/\-/, \"\", e);}{if ((\$3\$4 >= b) && (\$3\$4 < e)) print \$0}'"
else
	ssh guest@10.143.53.234 "cd /search/odin/softlink/qq; ls qq* | sort -t \"-\" -k3n -k4.1,4.2n | \
								awk -F'-|_' -v b=$BEGIN -v e=$END 'BEGIN{gsub(/\-/, \"\", b); gsub(/\-/, \"\", e);}{if ((\$3\$4 >= b) && (\$3\$4 < e)) print \$0}' | xargs cat "
fi
