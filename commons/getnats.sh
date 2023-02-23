# check if the `archive` directory exists
if [-d archive]; then 
    echo "archive folder exists"
    :
else
    echo "archive directory does not exist, creating it..."
    mkdir archive
fi

# download NATS Bench topological space to archive folder
wget --directory-prefix=archive/ https://www.dropbox.com/sh/ceeo70u1buow681/AADxyCvBAnE6mMjp7JOoo0LVa/NATS-tss-v1_0-3ffb9-simple.tar
# extract tar file and remove it
tar -xf archive/NATS-tss-v1_0-3ffb9-simple.tar && rm archive/NATS-tss-v1_0-3ffb9-simple.tar
# move the extracted NATS bench to archive folder
mv NATS-tss-v1_0-3ffb9-simple archive/NATS-tss-v1_0-3ffb9-simple