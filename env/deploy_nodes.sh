#!/bin/bash
DEST_DIR=/search/odin/offline/Workshop/servers
SERVER_DIR=/search/odin/Nick/GenerateWorkshop/servers

#models="attn-s2s-all-downsample-addmem"
#models="attn-bi-s2s-addmem-poem2 attn-bi-s2s-addmem-poem cvae-bi-simpleprior-attn-poem"
#models="vae-1024-attn-addmem"

cat nodes | while read line 
do
    if [[ ${line:0:1} = "#" ]]
    then
        continue
    fi
    DEST=${line%% *} 
    echo "Setup "$DEST"..."

    #echo "Copy tensorflow_serving binary..."
    #scp -r $SERVER_DIR/tensorflow_model_server $DEST:$DEST_DIR/

    echo "Copy tensorflow_serving deployments"
    for model in $models
    do
        #nohup scp -r $SERVER_DIR/deployments/$model $DEST:$DEST_DIR/deployments/ &
        scp -r $SERVER_DIR/deployments/$model $DEST:$DEST_DIR/deployments
    done
done
