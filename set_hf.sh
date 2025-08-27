if [ -z "$SCRATCH" ]; then
    echo "Error: SCRATCH environment variable is not set."
    exit 1
fi

if [ ! -d "$SCRATCH/.huggingface_cache" ]; then
    sudo mkdir "$SCRATCH/.huggingface_cache"
    echo "Folder created for huggingface cache at $SCRATCH/.huggingface_cache."
fi

export HF_HOME="$SCRATCH/.huggingface_cache"
echo "Set HF_HOME to $HF_HOME"

# Retrieve the primary group dynamically
MYGROUP=$(id -gn)

# Change ownership using the dynamically obtained group name
sudo chown -R $USER:$MYGROUP "$SCRATCH/.huggingface_cache"
echo "Added ${USER}:${MYGROUP} to $SCRATCH/.huggingface_cache"
